import logging
from dataclasses import dataclass

from dice.shared.modules import ModuleType

from .modules import ModuleImpl
from .repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class Signature:
    t: ModuleType
    name: str
    modules: list[ModuleImpl]

    def initialize(self, repo: Repository) -> "Signature":
        for m in self.modules:
            m.initialize(repo)
        return self

    def handle(self) -> None:
        for m in self.modules:
            m.run()

    def add(self, *module: ModuleImpl) -> "Signature":
        self.modules.extend(module)
        return self


def new_signature(t: ModuleType, name: str, *modules: ModuleImpl) -> Signature:
    return Signature(t, name, list(modules))
