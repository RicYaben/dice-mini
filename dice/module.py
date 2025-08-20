from dice.repo import Repository
from dice.models import Source
from dice.repo import load_repository

from collections.abc import Callable
from dataclasses import dataclass

M_CLASSIFIER: str = "classifier"
M_FINGERPRINTER: str = "fignerprinter"

type ModuleInit = Callable [[Repository], None]
type ModuleHandler = Callable[[Repository], None]

@dataclass
class Module:
    # type of module, classifier, fingerprinter, scanner...
    m_type: str
    # name of the module
    name: str
    
    # just take a connection and do something to it
    _init: ModuleInit
    _handler: ModuleHandler
    
    def init(self, repo: Repository) -> None:
        return self._init(repo)

    def handle(self, repo: Repository) -> None:
        return self._handler(repo)

@dataclass
class Signature:
    # type of signature
    s_type: str
    # name of the signature
    name: str
    # list of modules in the signature
    modules: list[Module]

    def init(self, repo: Repository) -> None:
        for m in self.modules:
            m.init(repo)

    def handle(self, repo: Repository) -> None:
        for m in self.modules:
            m.handle(repo)

@dataclass
class Component:
    # type of component: classifier, fingerprinter, scanner...
    c_type: str
    # name of the component
    name: str
    # list of signatures registered
    signatures: list[Signature]

    def init(self, repo: Repository) -> None:
        for s in self.signatures:
            s.init(repo)

    def handle(self, repo: Repository) -> None:
        for s in self.signatures:
            s.handle(repo)

@dataclass
class Engine:
    # list of components registered
    components: list[Component]

    def run(self, srcs: list[Source], db: str | None=None) -> Repository:
        # load all the sources
        repo = load_repository(srcs, db)

        for c in self.components:
            c.init(repo)
         
        # fingerprint hosts
        fps = [c for c in self.components if c.c_type == M_FINGERPRINTER]
        for fp in fps:
            fp.handle(repo)

        # classify hosts
        clss = [c for c in self.components if c.c_type == M_CLASSIFIER]
        for cl in clss:
            cl.handle(repo)

        return repo
    
def defaultModuleInit(_: Repository) -> None:
    pass
    
def new_module(t: str, name: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit) -> Module:
    return Module(t, name, handler, init)

def new_signature(t: str, name: str, *modules: Module) -> Signature:
    return Signature(t, name, list(modules))

def new_component(t: str, name: str, *signatures: Signature) -> Component:
    return Component(t, name, list(signatures))

def new_engine(*components: Component) -> Engine:
    return Engine(list(components))

@dataclass
class ComponentFactory:
    # type of component, signatures, and modules
    t: str
    name: str

    def make_signature(self, name: str, *module: Module) -> Signature:
        return new_signature(self.t, name, *module)
    
    def make_module(self, name: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit) -> Module:
        return new_module(self.t, name, handler, init)
    
    def make_component(self, *signature: Signature):
        return new_component(self.t, self.name, *signature)

def new_component_factory(t: str, name: str) -> ComponentFactory:
    return ComponentFactory(t, name)

def make_component(t: str, preffix: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit) -> Component:
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

def new_fingerprinter(handler: ModuleHandler, init: ModuleInit = defaultModuleInit, preffix: str = "cls") -> Component:
    return make_component(M_CLASSIFIER, preffix, handler, init)

def new_classifier(handler: ModuleHandler, init: ModuleInit = defaultModuleInit, preffix: str = "fp") -> Component:
    return make_component(M_FINGERPRINTER, preffix, handler, init)