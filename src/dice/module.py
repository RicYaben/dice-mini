from importlib import import_module
import pathlib
import json
import fnmatch
import sys
import pandas as pd

from typing import Optional, Generator, Any, Callable
from dataclasses import dataclass, field
from tabulate import tabulate
from tqdm import tqdm

from dice.config import CLASSIFIER, DEFAULT_BSIZE, FINGERPRINTER, MFACTORY, SCANNER, TAGGER, MType
from dice.query import query_db, query_records
from dice.repo import Repository, load_repository
from dice.models import Fingerprint, FingerprintLabel, HostTag, Source, Label, Tag
from dice.helpers import new_label, new_fp_label, new_fingerprint, new_tag, new_host_tag

type ModuleInit = Callable[['Module'], None]
type ModuleHandler = Callable[['Module'], None]
type RecordHandler = Callable[[Any], Generator[Any, None, None]]

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
        self._add = repo.save_models
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
    
    def make_handler(self, itemize, orient: str = 'records'):
        def records_handler(df: pd.DataFrame):
            items = []
            for r in df.itertuples(index=False):
                if item := itemize(r):
                    items.append(item)
            return items
        def batch_handler(df: pd.DataFrame):
            return itemize(df)
        return records_handler if orient == "records" else batch_handler

    def itemize(self, q: str, itemizer, orient: str= "records", pbar: bool=True) -> None:
        t, gen = self.repo().queryb(q)
        handler = self.make_handler(itemizer, orient)

        def with_pbar(total, batches):
            with tqdm(total=total, desc=f"{self.name}") as pbar:
                for b in batches:
                    items = handler(b)
                    self.save(*items)

                    pbar.update(len(b.index))
                    del b
 
        if pbar:
            with_pbar(t, gen)
            return
        
        for b in gen:
            items = handler(b)
            self.save(*items)
            del b

    def register_label(self, name: str, short: str="-", description: str="-", mitigation: str = "-"):
        lab = new_label(self.name, name, short,description, mitigation)
        self._labels[name] = lab
        self.repo().save_models(lab)

    def make_label(self, fp: str, lab: str) -> FingerprintLabel:
        l = self._labels[lab]
        return new_fp_label(fp, l.id)

    def label(self, fp: str, lab: str) -> None:
        flab = self.make_label(fp, lab)
        self.repo().save_models(flab)

    def make_fingerprint(self, rec: Any, data: dict, protocol: str="-") -> Fingerprint:
        data["probe_status"] = rec.get("data_status", "")
        return new_fingerprint(
            self.name, 
            rec.get("ip", "-"), 
            rec.get("id", "-"),
            json.dumps(data),
            protocol,
            rec.get("port", -1)
        )

    def register_tag(self, name: str, description: str="-") -> None:
        tag = new_tag(self.name, name, description)
        self._tags[name] = tag
        self.repo().save_models(tag)

    def make_tag(self, host: str, tag:str, details: str="", protocol: str="-", port: int= -1) -> HostTag:
        t = self._tags[tag]
        return new_host_tag(host, t.id, details, protocol, port)

    def make_fp_tag(self, fp, tag:str, details: str="") -> HostTag:
        return self.make_tag(fp["ip"], tag, details, fp["protocol"], fp["port"])
    
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

    def save(self, *items) -> None:
        return self.repo().save_models(*items)

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
        """
        Print engine info showing components, their type, signatures, and associated modules.
        One row per module, merging repeated Component / Type / Signature cells visually.
        """
        rows = []

        # collect rows: one row per module
        for comp in self.components:
            for sig in comp.signatures:
                for mod in sig.modules:
                    # get module path / collection if available
                    collection = getattr(mod, "collection", None)
                    module_name = f"{collection}:{mod.name}" if collection else mod.name
                    rows.append([
                        comp.name,
                        str(comp.c_type).upper(),
                        sig.name,
                        module_name
                    ])

        if not rows:
            print("No components found.")
            return

        # sort rows by Component, Type, Signature, Module
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

        # merge repeated cells visually
        last_comp = last_type = last_sig = None
        for row in rows:
            if row[0] == last_comp:
                row[0] = ""
            else:
                last_comp = row[0]

            if row[1] == last_type and row[0] == "":
                row[1] = ""
            else:
                last_type = row[1]

            if row[2] == last_sig and row[0] == "" and row[1] == "":
                row[2] = ""
            else:
                last_sig = row[2]

        print("\033[1mEngine information table.\033[0m  Includes loaded modules by components and signatures.")
        print(tabulate(rows, headers=["Component", "Type", "Signature", "Module"], tablefmt="rounded_outline"))
    
def defaultModuleInit(_) -> None:
    pass
    
def new_module(t: MType, name: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit) -> Module:
    return Module(t, name, init, handler)

# ---
# def new_plugin(
#         name: str, 
#         scanner: Module | None = None,
#         classifier: Module | None = None,
#         fingerprinter: Module | None = None,
#         tagger: Module | None = None,
#     ): raise NotImplementedError
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
        # registries registered
        self._registries: list[ModuleRegistry] = []
    
    def register(self, registry: 'ModuleRegistry') -> None:
        self._registries.append(registry)
    
    def find(self, modules: list[str] = ["*"]) -> list[tuple[str, Module]]:
        result: list[tuple[str, Module]] = []
        def matches_pattern(full_path_segments: list[str], pattern: str) -> bool:
            pat_segments = pattern.split(":")
            if len(pat_segments) == 1:
                # single segment: match any segment or module
                return any(fnmatch.fnmatch(seg, pat_segments[0]) for seg in full_path_segments)
            # multi-segment: check for sub-sequence match
            for i in range(len(full_path_segments) - len(pat_segments) + 1):
                if all(fnmatch.fnmatch(full_path_segments[i + j], pat_segments[j]) for j in range(len(pat_segments))):
                    return True
            return False

        def collect(registry: 'ModuleRegistry', path: list[str] = []):
            full_path = path + [registry.name]
            for m in registry.modules:
                full_path_with_module = full_path + [m.name]
                include = False
                if modules is None:
                    include = True
                else:
                    for pattern in modules:
                        if matches_pattern(full_path_with_module, pattern):
                            include = True
                            break
                if include:
                    result.append((":".join(full_path), m))
            for ch in registry.children.values():
                collect(ch, full_path)

        for reg in self._registries:
            collect(reg)

        return result
    
    def get_modules(self, t: MType | None = None, modules: list[str] = ["*"]) -> list[Module]:
        found = self.find(modules)
        found = [m for _,m in found if m.m_type == t]

        # Deduplicate
        uniq = {id(m): m for m in found}
        return list(uniq.values())
    
    def build(self, types: list[MType] = MFACTORY.all(), modules: list[str] = ["*"]) -> list[Component]:
        comps = []
        for t in types:
            if mods := self.get_modules(t, modules):
                signature = new_signature(t, self.name, *mods)
                c = new_component(t, self.name, signature)
                comps.append(c)
        return comps

    def info(self, modules: list[str] = ["*"]) -> None:
        modules = list(set(modules))
        found = self.find(modules=modules)

        if not found:
            print("No modules found.")
            return

        rows = [[path, str(m.m_type).capitalize(), m.name] for path, m in found]

        # sort and merge cells visually
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        last_collection = last_type = None
        for row in rows:
            if row[0] == last_collection:
                row[0] = ""
            else:
                last_collection = row[0]
            if row[1] == last_type and row[0] == "":
                row[1] = ""
            else:
                last_type = row[1]

        print("\033[1mRegistry information table.\033[0m  Includes matching available modules.")
        if modules != ["*"]:
            print(f'Queries: {", ".join(modules)}')
        print(tabulate(rows, headers=["Collection", "Type", "Module"], tablefmt="rounded_outline"))

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
            repo.save_models(*fps)
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
            repo.save_models(*labs)
        q = query_db("fingerprints", protocol=protocol)
        mod.with_pbar(handler, q)
    return wrapper

class ModuleRegistry:
    def __init__(self, name: str = "custom") -> None:
        self.name = name
        self.modules: list[Module] = []
        self.children: dict[str, 'ModuleRegistry'] = {}

    def add(self, *module: Module) -> 'ModuleRegistry':
        self.modules.extend(module)
        return self
    
    def add_group(self, group: 'ModuleRegistry', name: str | None = None) -> 'ModuleRegistry':
        if not name:
            name = group.name
        self.children[name] = group
        return self
    
    def add_groups(self, groups: list['ModuleRegistry']) -> 'ModuleRegistry':
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