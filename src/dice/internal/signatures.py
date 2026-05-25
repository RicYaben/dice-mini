from dataclasses import dataclass

from .repository import Repository
from .modules import Module

from dice.shared.modules import ModuleType

import logging

logger = logging.getLogger(__name__)


@dataclass
class Signature:
    t: ModuleType
    name: str
    modules: list[Module]

    def initialize(self, repo: Repository) -> "Signature":
        for m in self.modules:
            m.initialize(repo)
        return self

    def handle(self) -> None:
        for m in self.modules:
            m.run()

    def add(self, *module: Module) -> "Signature":
        self.modules.extend(module)
        return self


def new_signature(t: ModuleType, name: str, *modules: Module) -> Signature:
    return Signature(t, name, list(modules))
