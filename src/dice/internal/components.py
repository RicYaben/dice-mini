import fnmatch
import logging
from dataclasses import dataclass

from tabulate import tabulate

from dice.shared.modules import MFACTORY, ModuleType

from .config import Configuration
from .modules import (
    Module,
    ModuleRegistry,
)
from .repository import Repository
from .signatures import Signature, new_signature

logger = logging.getLogger(__name__)

# TODO: we can make components and signatures a unique interface object with children
@dataclass
class Component:
    # type of component: classifier, fingerprinter, scanner...
    t: ModuleType
    name: str
    signatures: list[Signature]

    @property
    def modules(self) -> list[Module]:
        mods = []
        for s in self.signatures:
            mods.extend(s.modules)
        return mods

    def initialize(self, repo: Repository) -> "Component":
        for s in self.signatures:
            s.initialize(repo)
        return self

    def handle(self) -> None:
        for s in self.signatures:
            s.handle()

    def add(self, *signature: Signature) -> "Component":
        self.signatures.extend(signature)
        return self

    def registrys(self) -> set[str]:
        return {mod.registry for mod in self.modules if mod.registry}

    def to_dict(self) -> dict:
        return {
            "type": str(self.t),
            "name": self.name,
            "modules": [m.to_dict() for m in self.modules],
        }

def new_component(t: ModuleType, name: str, *signatures: Signature) -> Component:
    return Component(t, name, list(signatures))

class Components:
    def __init__(self, comps: list[Component]) -> None:
        self._comps = comps

    @property
    def modules(self) -> list[Module]:
        mods = []
        for c in self._comps:
            mods.extend(c.modules)
        return mods

    def add(self, *comps: Component) -> "Components":
        self._comps.extend(comps)
        return self

    def extend(self, comps: "Components") -> "Components":
        self._comps.extend(comps._comps)
        return self

    def configure(self, config: Configuration) -> "Components":
        for mod in self.modules:
            if f := config.data[mod.desc.name]:
                mod.desc.flags.update(**f)
        return self

    def initialize(self, repo: Repository) -> "Components":
        for c in self._comps:
            c.initialize(repo)
        return self

    def handle(self) -> None:
        for c in self._comps:
            c.handle()

    def registrys(self) -> list[str]:
        regs = set()
        for c in self._comps:
            r = c.registrys()
            regs.update(r)
        return list(regs)

    def to_dict(self) -> list[dict]:
        return [c.to_dict() for c in self._comps]

    def __repr__(self) -> str:
        return str(self.to_dict())


class ComponentManager:
    def __init__(self, name: str = "comp") -> None:
        self.name = name
        self._registries: list[ModuleRegistry] = []

    def register(self, registry: "ModuleRegistry") -> 'ComponentManager':
        self._registries.append(registry)
        return self

    def find(self, modules: list[str] | None = None) -> list[tuple[str, Module]]:
        def matches_pattern(full_path_segments: list[str], pattern: str) -> bool:
            pat_segments = pattern.split(":")
            if len(pat_segments) == 1:
                # single segment: match any segment or module
                return any(
                    fnmatch.fnmatch(seg, pat_segments[0]) for seg in full_path_segments
                )
            # multi-segment: check for sub-sequence match
            for i in range(len(full_path_segments) - len(pat_segments) + 1):
                if all(
                    fnmatch.fnmatch(full_path_segments[i + j], pat_segments[j])
                    for j in range(len(pat_segments))
                ):
                    return True
            return False

        def collect(mods: list[str], registry: "ModuleRegistry", path: list[str]) -> list[tuple[str, Module]]:
            result = []
            path.append(registry.name)

            for m in registry.modules:
                fpath_mod = path + [m.desc.name]

                for pattern in mods:
                    if matches_pattern(fpath_mod, pattern):
                        p = ":".join(path)
                        m.registry = p
                        result.append((p, m))
                        break

            for ch in registry.children.values():
                result.extend(collect(mods, ch, path))

            return result

        if not modules:
            modules = ["*"]

        result = []
        for reg in self._registries:
            result.extend(collect(modules, reg, []))

        return result

    def get_modules(
        self, t: ModuleType | None = None, modules: list[str] | None = None
    ) -> list[Module]:
        if not modules:
            modules = ["*"]

        found = self.find(modules)
        found = [m for _, m in found if m.t == t]

        # Deduplicate
        uniq = {id(m): m for m in found}
        return list(uniq.values())

    def build(
        self, types: list[ModuleType] | None = None, modules: list[str] | None = None
    ) -> Components:
        if not modules:
            modules = ["*"]
        comps = []
        for t in types or MFACTORY.all():
            if c := self.make(t, modules):
                comps.append(c)
        return Components(comps)

    def make(self, type: ModuleType, modules: list[str] | None = None) -> Component | None:
        if not modules:
            modules = ["*"]

        if not (mods := self.get_modules(type, modules)):
            return None
        signature = new_signature(type, self.name, *mods)
        return new_component(type, self.name, signature)

    def info(self, modules: list[str] | None = None) -> None:
        if not modules:
            modules = ["*"]

        modules = list(set(modules))
        found = self.find(modules=modules)

        if not found:
            logger.info("No modules found.")
            return

        rows = [[path, str(m.t).capitalize(), m.desc.name] for path, m in found]

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

        logger.info(
            "\033[1mRegistry information table.\033[0m  Includes matching available modules."
        )
        if modules != ["*"]:
            logger.info(f"Queries: {', '.join(modules)}")

        msg = tabulate(
            rows,
            headers=["Collection", "Type", "Module"],
            tablefmt="rounded_outline",
        )
        logger.info(f"\n{msg}")


def new_component_manager(study: str) -> ComponentManager:
    return ComponentManager(study)
