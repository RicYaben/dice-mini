import fnmatch
import logging
import pathlib
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points

from dice.shared.modules import ModuleDescriptor, ModuleEnum, ModuleType, find_module
from dice.shared.repository import BaseRepo, CRepo, FRepo, TRepo

from .repository import Repository

logger = logging.getLogger(__name__)


def make_repository(base: Repository, name: str, t: ModuleType) -> BaseRepo:
    r = None
    match t:
        case ModuleEnum.FINGERPRINTER.value:
            r = FRepo
        case ModuleEnum.CLASSIFIER.value:
            r = CRepo
        case ModuleEnum.TAGGER.value:
            r = TRepo
        case _:
            r = BaseRepo
    return r(base, name)


@dataclass
class Module:
    t: ModuleType
    desc: ModuleDescriptor

    def initialize(self, repo: Repository) -> None:
        if tags := self.desc.tags:
            repo.insert(items=tags)

        if labs := self.desc.labels:
            repo.insert(items=labs)

        logger = logging.getLogger(self.desc.name)
        drepo = make_repository(repo, self.desc.name, self.t)
        self.desc.initialize(drepo, logger)
        return self.desc.pre()

    def run(self) -> None:
        return self.desc.run()

    def post(self) -> None:
        return self.desc.post()


class ModuleRegistry:
    def __init__(self, name: str = "custom") -> None:
        self.name = name
        self.modules: list[Module] = []
        self.children: dict[str, ModuleRegistry] = {}

    def register(self, desc: ModuleDescriptor) -> "ModuleRegistry":
        t = find_module(desc.t)
        mod = Module(t, desc)
        self.add(mod)
        return self

    def add(self, *modules: Module) -> "ModuleRegistry":
        self.modules.extend(modules)
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
            return [m for m in self.modules if fnmatch.fnmatch(m.desc.name, mod)]

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
