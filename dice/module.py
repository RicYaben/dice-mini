from dice.repo import Repository
from dice.models import Source
from dice.repo import load_repository

from collections.abc import Callable

type ModuleHandler = Callable[[Repository], None]

class Module:
    # type of module, classifier, fingerprinter, scanner...
    m_type: str
    # name of the module
    name: str
    
    # just take a connection and do something to it
    _handler: ModuleHandler

    def handle(self, repo: Repository) -> None:
        return self._handler(repo)


class Signature:
    # type of signature
    s_type: str
    # name of the signature
    name: str
    # list of modules in the signature
    modules: list[Module]
    # tags to filter which signatures to use
    tags: list[str]

    def handle(self, repo: Repository) -> None:
        for m in self.modules:
            m.handle(repo)


class Component:
    # type of component: classifier, fingerprinter, scanner...
    c_type: str
    # name of the component
    name: str
    # list of signatures registered
    signatures: list[Module]
    # list of tags to filter components
    tags: list[str]

    def handle(self, repo: Repository) -> None:
        for s in self.signatures:
            s.handle(repo)


class Engine:
    # list of components registered
    components: list[Component]

    def run(self, srcs: list[Source], db: str | None=None) -> Repository:
        # load all the sources
        repo = load_repository(srcs, db)
         
        # fingerprint hosts
        fps = [c for c in self.components if c.c_type == "fingerprinter"]
        for fp in fps:
            fp.handle(repo)

        # classify hosts
        clss = [c for c in self.components if c.c_type == "classifier"]
        for cl in clss:
            cl.handle(repo)

        return repo