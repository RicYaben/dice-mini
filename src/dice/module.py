from importlib.util import spec_from_file_location, module_from_spec
from importlib import import_module
import pathlib
import json
import fnmatch
import sys
import pandas as pd

from typing import Optional, Generator, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm

from dice.config import CLASSIFIER, DEFAULT_BSIZE, FINGERPRINTER, MFACTORY, SCANNER, TAGGER, MType
from dice.query import query_db, query_records
from dice.repo import Repository, load_repository
from dice.models import Fingerprint, FingerprintLabel, HostTag, Source, Label, Tag
from dice.helpers import new_label, new_fp_label, new_fingerprint, new_tag, new_host_tag

type ModuleInit = Callable[['Module'], None]
type ModuleHandler = Callable[['Module'], None]

@dataclass
class Module:
    # type of module, classifier, fingerprinter, scanner...
    m_type: MType
    # name of the module
    name: str
    
    # just take a connection and do something to it
    _init: ModuleInit
    _handler: ModuleHandler
    _repo: Optional[Repository] = None

    # registered labels
    _labels: dict[str,Label] = field(default_factory=dict)
    _tags: dict[str,Tag] = field(default_factory=dict)

    # cache
    _cache: dict[str, Any] = field(default_factory=dict)
    
    def init(self, repo: Repository) -> None:
        self._repo = repo
        return self._init(self)

    def handle(self) -> None:
        print(f"[{self.name}]")
        return self._handler(self)
    
    def repo(self) -> Repository:
        if not self._repo:
            raise Exception("repository not set")
        return self._repo
    
    def query(self, q: str) -> Generator[Any, None, None]:
        return self.repo().query(q)
    
    def register_label(self, name: str, short: str="-", description: str="-", mitigation: str = "-"):
        lab = new_label(self.name, name, short,description, mitigation)
        self._labels[name] = lab
        self.repo().add_labels(lab)

    def make_label(self, fp: str, lab: str) -> FingerprintLabel:
        l = self._labels[lab]
        return new_fp_label(fp, l.id)

    def label(self, fp: str, lab: str) -> None:
        flab = self.make_label(fp, lab)
        self.repo().label(flab)

    def make_fingerprint(self, rec: Any, data: dict, protocol: str="-") -> Fingerprint:
        return new_fingerprint(
            self.name, 
            rec.get("ip", "-"), 
            rec.get("id", "-"),
            json.dumps(data),
            protocol,
            rec.get("port", -1)
        )

    def fingerprint(self, rec: pd.Series, data: dict, protocol: str="-"):
        fp = self.make_fingerprint(rec, data, protocol)
        self.repo().fingerprint(fp)

    def register_tag(self, name: str, description: str="-") -> None:
        tag = new_tag(self.name, name, description)
        self._tags[name] = tag
        self.repo().add_tags(tag)

    def tag(self, host: str, tag: str, details: str="", protocol: str="-", port: int= -1) -> None:
        htag = self.make_tag(host, tag, details, protocol, port)
        self.repo().tag(htag)

    def make_tag(self, host: str, tag:str, details: str="", protocol: str="-", port: int= -1) -> HostTag:
        t = self._tags[tag]
        return new_host_tag(host, t.id, details, protocol, port)

    def tag_fp(self, fp, tag:str, details: str="") -> None:
        return self.tag(fp["ip"], tag, details, fp["protocol"], fp["port"])
    
    def cache(self, name: str, value: Any) -> None:
        self._cache[name] = value
    
    def get_cache(self, name: str) -> Any:
        return self._cache[name]
    
    def with_pbar(self, handler: Callable[[pd.DataFrame], None], q: str, bsize: int = DEFAULT_BSIZE) -> None:
        t, gen = self.repo().queryb(q)
        with tqdm(total=t, desc=f"{self.name}") as pbar:
            pbar.write(f"fetching in batches ({bsize}/b)")
            for b in gen:
                handler(b)
                pbar.update(len(b.index))
                del b

@dataclass
class Signature:
    # type of signature
    s_type: MType
    # name of the signature
    name: str
    # list of modules in the signature
    modules: list[Module]

    def init(self, repo: Repository) -> 'Signature':
        for m in self.modules:
            m.init(repo)
        return self

    def handle(self) -> None:
        for m in self.modules:
            m.handle()

    def add(self, *module: Module) -> 'Signature':
        self.modules.extend(module)
        return self

@dataclass
class Component:
    # type of component: classifier, fingerprinter, scanner...
    c_type: MType
    # name of the component
    name: str
    # list of signatures registered
    signatures: list[Signature]

    def init(self, repo: Repository) -> 'Component':
        for s in self.signatures:
            s.init(repo)
        return self

    def handle(self) -> None:
        for s in self.signatures:
            s.handle()

    def add(self, *signature: Signature) -> 'Component':
        self.signatures.extend(signature)
        return self

@dataclass
class Engine:
    # list of components registered
    components: list[Component]

    def run(self, srcs: list[Source] = [], db: str | None =None, repo: Repository | None = None) -> Repository:
        # load all the sources
        if not repo:    
            repo = load_repository(db=db)
        repo.add_sources(*srcs)
        
        def fcomp(t):
            return lambda c: c.c_type == t

        print("initializing")
        for c in self.components:
            c.init(repo)

        print("shaking vigorously")
        for m in [SCANNER, FINGERPRINTER, CLASSIFIER, TAGGER]:
            if comps := list(filter(fcomp(m), self.components)):
                print(f"rolling {m.name}(s)")
                for c in comps:
                    c.handle()

        return repo
    
    def info(self) -> None:
        rows = []
        for comp in self.components:
            for sig in comp.signatures:
                rows.append({
                    "Component": comp.name,
                    "Type": str(comp.c_type).upper(),
                    "Signature": sig.name,
                    "Modules": ", ".join(sorted(set([mod.name for mod in sig.modules]))),
                })

        print("Engine Info:")
        print(tabulate(rows, headers="keys", tablefmt="github"))
    
def defaultModuleInit(_) -> None:
    pass
    
def new_module(t: MType, name: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit) -> Module:
    return Module(t, name, init, handler)

# ---
def new_plugin(
        name: str, 
        scanner: Module | None = None,
        classifier: Module | None = None,
        fingerprinter: Module | None = None,
        tagger: Module | None = None,
    ): raise NotImplementedError
# ---

def new_signature(t: MType, name: str, *modules: Module) -> Signature:
    return Signature(t, name, list(modules))

def new_component(t: MType, name: str, *signatures: Signature) -> Component:
    return Component(t, name, list(signatures))

def new_engine(*components: Component) -> Engine:
    return Engine(list(components))

@dataclass
class ComponentFactory:
    # type of component, signatures, and modules
    t: MType
    name: str

    def make_signature(self, name: str, *module: Module) -> Signature:
        return new_signature(self.t, name, *module)
    
    def make_module(self, name: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit) -> Module:
        return new_module(self.t, name, handler, init)
    
    def make_component(self, *signature: Signature) -> Component:
        return new_component(self.t, self.name, *signature)

def new_component_factory(t: MType, name: str) -> ComponentFactory:
    return ComponentFactory(t, name)

def make_component(t: MType, preffix: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit) -> Component:
    fact = new_component_factory(t, "-".join([preffix, "comp"]))
    return fact.make_component(
        fact.make_signature(
            "-".join([preffix, "sig"]), 
            fact.make_module(
                "-".join([preffix, "mod"]),
                handler,
                init
            )
        )
    )

def new_fingerprinter(handler: ModuleHandler, init: ModuleInit = defaultModuleInit, preffix: str = "fp") -> Component:
    return make_component(FINGERPRINTER, preffix, handler, init)

def new_classifier(handler: ModuleHandler, init: ModuleInit = defaultModuleInit, preffix: str = "cls") -> Component:
    return make_component(CLASSIFIER, preffix, handler, init)

class ComponentManager:
    def __init__(self, name: str = "comp") -> None:
        self.name = name
        self._modules: list[Module] = []

    def add(self, *module: Module) -> 'ComponentManager':
        self._modules.extend(module)
        return self

    def build(self, types: list[MType] = MFACTORY.all(), modules: list[str] = ["*"]) -> list[Component]:
        comps = []
        for t in types:
            if mods := self.get_modules(t, modules):
                signature = new_signature(t, self.name, *mods)
                c = new_component(t, self.name, signature)
                comps.append(c)
        return comps

    def get_modules(self, t: MType | None = None, modules: list[str]=["*"]) -> list[Module]:
        m: list[Module] = []

        fmods = self._modules
        if t:
            fmods = list(filter(lambda m: m.m_type == t, self._modules))

        for mod in fmods:
            for p in modules:
                if fnmatch.fnmatch(mod.name, p):
                    if mod not in m:
                        m.append(mod)
                    break
        return list(m)

    def info(self) -> None:
        mods = defaultdict(list)
        for mod in self._modules:
            mods[str(mod.m_type).upper()].append(mod.name)

        print("Registered modules:")
        print(tabulate(mods, headers="keys", tablefmt="rounded_outline"))

    def list_modules(self, modules: list[str] = ["*"]) -> None:
        mods = defaultdict(list)
        for m in self.get_modules(modules=modules):
            mods[str(m.m_type).upper()].append(m.name)

        print(tabulate(mods, headers="keys", tablefmt="rounded_outline"))

def new_component_manager(study: str) -> ComponentManager:
    return ComponentManager(study)

def make_fp_handler(fp_cb: Callable[[pd.Series], dict | None], protocol: str="-", source: str = "zgrab2") -> ModuleHandler:
    def wrapper(mod: Module) -> None:
        repo = mod.repo()
        def handler(df: pd.DataFrame):
            fps = []
            for _,r in df.iterrows():
                if fp := fp_cb(r):
                    fps.append(mod.make_fingerprint(r, fp, protocol))
            repo.fingerprint(*fps)
        q = query_records(source=source, protocol=protocol)
        mod.with_pbar(handler, q)
    return wrapper

def make_cls_handler(cls_cb: Callable[[pd.Series], str | None], protocol="-") -> ModuleHandler:
    def wrapper(mod: Module) -> None:
        repo = mod.repo()
        def handler(df: pd.DataFrame):
            labs = []
            for _,fp in df.iterrows():
                if lab := cls_cb(fp):
                    labs.append(mod.make_label(str(fp["id"]), lab))
            repo.label(*labs)
        q = query_db("fingerprints", protocol=protocol)
        mod.with_pbar(handler, q)
    return wrapper

class ModuleRegistry:
    def __init__(self, name: str = "custom") -> None:
        self.name = name
        self._modules = []

    def add(self, *module: Module) -> 'ModuleRegistry':
        self._modules.extend(module)
        return self
    
    def all(self) -> list[Module]:
        return self._modules
    
def new_registry(name: str) -> ModuleRegistry:
    registry = ModuleRegistry(name)
    return registry

def load_registry(p: str):
    pp = pathlib.Path(p).resolve()
    sys.path.insert(0, str(pp.parent))  # parent of modules
    
    registry = import_module(pp.name)
    return registry.REGISTRY