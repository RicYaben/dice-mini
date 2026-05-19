from typing import Literal, Optional, Generator, Any, Callable, Type
from dataclasses import dataclass, field
from tqdm import tqdm
from importlib import import_module
from importlib.metadata import entry_points

import pathlib
import json
import fnmatch
import sys
import pandas as pd
import logging

from dice.config import (
    DEFAULT_BSIZE,
    ModuleType,
    find_module
)

from dice.database import insert_or_ignore
from dice.helpers import get_normalizer
from dice.query import query_db, query_records
from dice.repo import Repository
from dice.models import Fingerprint, FingerprintLabel, HostTag, Label, Tag
from dice.constructors import (
    new_label,
    new_fp_label,
    new_fingerprint,
    new_tag,
    new_host_tag,
)

logger = logging.getLogger(__name__)


type ModuleInit = Callable[["Module"], None]
type ModuleHandler = Callable[["Module"], None]
type RecordHandler = Callable[[Any], Generator[Any, None, None]]


@dataclass
class Module:
    # type of module, classifier, fingerprinter, scanner...
    m_type: ModuleType
    # name of the module
    name: str

    _init: ModuleInit
    _handler: ModuleHandler
    _repo: Optional[Repository] = None

    collection: Optional[str] = None

    # registered labels
    _labels: dict[str, Label] = field(default_factory=dict)
    _tags: dict[str, Tag] = field(default_factory=dict)

    # cache, temporarely stores items
    _model: Optional[Type] = None
    _cache: list = field(default_factory=list)
    _temp_size: int = DEFAULT_BSIZE

    def init(self, repo: Repository) -> None:
        self._repo = repo
        return self._init(self)

    def flush(self) -> None:
        "flushes remaining items in the cache"
        if not self._cache:
            return
        self.repo().insert(self._cache)

    def handle(self) -> None:
        logger.info(f"[{self.name}]")
        self._handler(self)
        self.flush()

    def repo(self) -> Repository:
        if not self._repo:
            raise Exception("repository not set")
        return self._repo

    def query(self, q: str) -> Generator[Any, None, None]:
        return self.repo().simple_query(q)

    # ----
    def register_label(
        self,
        name: str,
        short: str = "-",
        description: str = "-",
        mitigation: str = "-",
        level=0,
    ):
        lab = new_label(self.name, name, short, description, mitigation, level)
        self._labels[name] = lab
        self.repo().insert([lab])

    def register_tag(self, name: str, description: str = "-") -> None:
        tag = new_tag(self.name, name, description)
        self._tags[name] = tag
        self.repo().insert([tag])

    # ----

    def make_label(self, fp: int, lab: str, comment: str = "") -> FingerprintLabel:
        slab = self._labels[lab]
        return new_fp_label(fp, slab.id) # type: ignore

    def make_fingerprint(
        self, rec: Any, data: dict, protocol: str = "-"
    ) -> Fingerprint:
        data["probe_status"] = rec.get("data_status", "")
        return new_fingerprint(
            self.name,
            rec.get("ip", "-"),
            rec.get("id", "-"),
            rec.get("resource_id", "-"),
            json.dumps(data),
            protocol,
            rec.get("port", -1),
        )

    def make_tag(
        self,
        host: str,
        tag: str,
        details: str = "",
        protocol: str = "-",
        port: int = -1,
    ) -> HostTag:
        t = self._tags[tag]
        return new_host_tag(host, t.id, details, protocol, port) # type: ignore

    def make_fp_tag(self, fp, tag: str, details: str = "") -> HostTag:
        return self.make_tag(fp["host"], tag, details, fp["protocol"], fp["port"])

    # ---

    def with_pbar(
        self,
        handler: Callable[[pd.DataFrame], None],
        q: str,
        bsize: int = DEFAULT_BSIZE,
        desc: str | None = None,
    ) -> None:
        self._temp_size = bsize
        if not desc:
            desc = self.name

        t, gen = self.repo().query(q, bsize=bsize)
        with tqdm(total=t, desc=desc) as pbar:
            pbar.write(f"fetching in batches ({bsize}/b)")
            for b in gen:
                size = len(b.index)
                handler(b)
                pbar.update(size)
                del b

        self.flush()
        self._temp_size = DEFAULT_BSIZE

    def store(self, *items) -> None:
        self._cache.extend(items)
        if len(self._cache) >= self._temp_size:
            self.flush()

    def itemize(
        self,
        q: str,
        itemizer,
        orient: Literal["rows", "tuples", "dataframe"] = "rows",
        pbar: bool = True,
        norm = None,
    ) -> None:
        """
        Orient:
        - rows: a pd.series
        - tuples: a pandas object
        - dataframe: the whole df
        """

        def iter_orient(orient: str):
            match orient:
                case "tuples":

                    def tup(x: pd.DataFrame):
                        for t in x.itertuples(index=False):
                            itemizer(t)

                    return tup
                case "rows":

                    def r(x: pd.DataFrame):
                        for _, t in x.iterrows():
                            itemizer(t)

                    return r
                case "dataframe":
                    return lambda x: itemizer(x)
                case _:
                    raise ValueError(f"unknown orientation: {orient}")

        t, gen = self.repo().query(q, norm=norm)
        it = iter_orient(orient)
        if not pbar:
            for b in gen:
                it(b)
                del b

        with tqdm(total=t, desc=f"{self.name}") as bar:
            for b in gen:
                n = len(b.index)
                it(b)
                bar.update(n)
                del b


def defaultModuleInit(_) -> None:
    pass


def new_module(
    t: ModuleType | str, name: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit
) -> Module:
    if isinstance(t, str):
        t = find_module(t)
    return Module(t, name, init, handler)


class ModuleRegistry:
    def __init__(self, name: str = "custom") -> None:
        self.name = name
        self.modules: list[Module] = []
        self.children: dict[str, "ModuleRegistry"] = {}

    def add(self, *module: Module) -> "ModuleRegistry":
        for mod in module:
            mod.collection = self.name

        self.modules.extend(module)
        return self

    def add_group(
        self, group: "ModuleRegistry", name: str | None = None
    ) -> "ModuleRegistry":
        if not name:
            name = group.name
        self.children[name] = group
        return self

    def add_groups(self, groups: list["ModuleRegistry"]) -> "ModuleRegistry":
        for g in groups:
            self.add_group(g)
        return self

    def all(self) -> list[Module]:
        mods = []
        for g in self.children.values():
            mods.extend(g.all())
        mods.extend(self.modules)
        return mods

    def find(self, path: str) -> list[Module]:
        parts = path.split(":", 1)

        # If only registry is queried ("honeypots")
        if len(parts) == 1:
            if fnmatch.fnmatch(self.name, parts[0]):
                return self.all()
            # otherwise search children
            mods = []
            for ch in self.children.values():
                mods.extend(ch.find(path))
            return mods

        # If registry + module ("honeypots:cowrie")
        reg, mod = parts
        if fnmatch.fnmatch(self.name, reg):
            return [m for m in self.modules if fnmatch.fnmatch(m.name, mod)]

        # otherwise recurse
        mods = []
        for ch in self.children.values():
            mods.extend(ch.find(path))
        return mods


def new_registry(name: str) -> ModuleRegistry:
    registry = ModuleRegistry(name)
    return registry


def load_registry(p: str):
    pp = pathlib.Path(p).resolve()
    sys.path.insert(0, str(pp.parent))  # parent of modules

    registry = import_module(pp.name)
    return registry.registry


def load_registry_plugins(group: str) -> list[ModuleRegistry]:
    groups = []
    plugins = entry_points(group=group)
    for ep in plugins:
        registry = ep.load()
        groups.append(registry)
    return groups


FPCallback = Callable[[pd.Series], dict | None]
RowHandler = Callable[[pd.Series], None]

def default_handler(
    mod: Module,
    fp_cb: FPCallback,
    protocol: str,
) -> RowHandler:
    def handler(r: pd.Series):
        if fp := fp_cb(r):
            mod.store(mod.make_fingerprint(r, fp, protocol))
    return handler

def zgrab2_handler(
    mod: Module,
    fp_cb: FPCallback,
    protocol: str,
) -> RowHandler:

    def handler(r: pd.Series):
        # TODO: this in the future
        # is_proto = eval_communication(r), # true or false

        # # return early, is a false-positive
        # if not is_proto:
        #     return

        # base = {
        #     "is_protocol": is_proto,
        #     "connection": eval_status(r), # connected, refused
        #     "encryption": eval_encryption(r), # TLS, DTLS, or whatever other scheme; otherwise None
        #     "certificates": r.get("data_certificates", None)
        # }

        if fp := fp_cb(r):
            #base.update(fp)
            mod.store(mod.make_fingerprint(r, fp, protocol))
    return handler

def make_fp_handler(
    fp_cb: FPCallback,
    protocol: str = "-",
    source: str = "zgrab2",
) -> ModuleHandler:
    def wrapper(mod: Module) -> None:
        match source:
            case "zgrab2":
                h = zgrab2_handler(mod, fp_cb, protocol)
            case _:
                h = default_handler(mod, fp_cb, protocol)

        q = query_records(source=source, protocol=protocol)
        norm = get_normalizer(source)
        mod.itemize(q, h, orient="rows", norm=norm)
    return wrapper

def make_cls_handler(
    cls_cb: Callable[[pd.Series], str | None], protocol="-"
) -> ModuleHandler:
    def wrapper(mod: Module) -> None:
        repo = mod.repo()

        def handler(df: pd.DataFrame):
            labs = []
            for _, fp in df.iterrows():
                if lab := cls_cb(fp):
                    labs.append(mod.make_label(fp["id"].hex, lab))

            if not labs:
                return

            with repo.session() as ses:
                insert_or_ignore(ses, FingerprintLabel, labs)

        q = query_db("fingerprint", protocol=protocol)
        mod.with_pbar(handler, q)

    return wrapper