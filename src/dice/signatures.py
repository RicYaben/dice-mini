from dataclasses import dataclass

import logging

from dice.config import MType
from dice.repo import Repository
from dice.modules import Module

logger = logging.getLogger(__name__)

@dataclass
class Signature:
    # type of signature
    s_type: MType
    # name of the signature
    name: str
    # list of modules in the signature
    modules: list[Module]

    def init(self, repo: Repository) -> "Signature":
        for m in self.modules:
            m.init(repo)
        return self

    def handle(self) -> None:
        for m in self.modules:
            m.handle()

    def add(self, *module: Module) -> "Signature":
        self.modules.extend(module)
        return self
    
def new_signature(t: MType, name: str, *modules: Module) -> Signature:
    return Signature(t, name, list(modules))