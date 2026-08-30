import logging
from dataclasses import dataclass

from dice.shared.flags import Flags
from dice.shared.models import Label, Tag
from dice.shared.modules import ModuleDescriptor, Runner, do_nothing
from dice.shared.repository import BaseRepo


def wrap_runner(msg: str, runner: Runner) -> Runner:
    if runner == do_nothing:
        return runner

    def wrapper(repo: BaseRepo, flags: Flags, logger: logging.Logger) -> None:
        logger.info(msg)
        return runner(repo, flags, logger)

    return wrapper


@dataclass
class Service:
    name: str
    vendor: str | None
    version: str | None
    cpe: str | None


@dataclass
class Module(ModuleDescriptor):
    def __post_init__(self):
        self.run_fn = wrap_runner("starting", self.run_fn)
        self.pre_fn = wrap_runner("initializing", self.pre_fn)
        self.post_fn = wrap_runner("cleaning", self.post_fn)

    def add_label(self, name: str, description: str | None = None) -> "Module":
        self.labels.append(
            Label(name=name, description=description, module_name=self.name)
        )
        return self

    def add_tag(self, name: str, description: str) -> "Module":
        self.tags.append(Tag(name=name, description=description, module_name=self.name))
        return self
