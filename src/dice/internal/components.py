import fnmatch
import logging

from dataclasses import dataclass
from tabulate import tabulate

from .repository import Repository
from .signatures import Signature, new_signature
from .modules import (
    Module,
    ModuleRegistry,
)

from dice.shared.modules import MFACTORY, ModuleType

logger = logging.getLogger(__name__)


@dataclass
class Component:
    # type of component: classifier, fingerprinter, scanner...
    t: ModuleType
    # name of the component
    name: str
    # list of signatures registered
    signatures: list[Signature]

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


def new_component(t: ModuleType, name: str, *signatures: Signature) -> Component:
    return Component(t, name, list(signatures))


class ComponentManager:
    def __init__(self, name: str = "comp") -> None:
        self.name = name
        # registries registered
        self._registries: list[ModuleRegistry] = []

    def register(self, registry: "ModuleRegistry") -> None:
        self._registries.append(registry)

    def find(self, modules: list[str] = ["*"]) -> list[tuple[str, Module]]:
        result: list[tuple[str, Module]] = []

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

        def collect(registry: "ModuleRegistry", path: list[str] = []):
            full_path = path + [registry.name]
            for m in registry.modules:
                full_path_with_module = full_path + [m.desc.name]
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

    def get_modules(
        self, t: ModuleType | None = None, modules: list[str] = ["*"]
    ) -> list[Module]:
        found = self.find(modules)
        found = [m for _, m in found if m.t == t]

        # Deduplicate
        uniq = {id(m): m for m in found}
        return list(uniq.values())

    def build(
        self, types: list[ModuleType] = MFACTORY.all(), modules: list[str] = ["*"]
    ) -> list[Component]:
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
