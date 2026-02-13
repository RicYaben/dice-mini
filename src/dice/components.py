import fnmatch
import logging

from dataclasses import dataclass
from tabulate import tabulate

from dice.config import MFACTORY, MType
from dice.repo import Repository
from dice.signatures import Signature, new_signature
from dice.modules import Module, ModuleHandler, ModuleInit, defaultModuleInit, new_module, ModuleRegistry

logger = logging.getLogger(__name__)

@dataclass
class Component:
    # type of component: classifier, fingerprinter, scanner...
    c_type: MType
    # name of the component
    name: str
    # list of signatures registered
    signatures: list[Signature]

    def init(self, repo: Repository) -> "Component":
        for s in self.signatures:
            s.init(repo)
        return self

    def handle(self) -> None:
        for s in self.signatures:
            s.handle()

    def add(self, *signature: Signature) -> "Component":
        self.signatures.extend(signature)
        return self


def new_component(t: MType, name: str, *signatures: Signature) -> Component:
    return Component(t, name, list(signatures))

@dataclass
class ComponentFactory:
    # type of component, signatures, and modules
    t: MType
    name: str

    def make_signature(self, name: str, *module: Module) -> Signature:
        return new_signature(self.t, name, *module)

    def make_module(
        self, name: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit
    ) -> Module:
        return new_module(self.t, name, handler, init)

    def make_component(self, *signature: Signature) -> Component:
        return new_component(self.t, self.name, *signature)


def new_component_factory(t: MType, name: str) -> ComponentFactory:
    return ComponentFactory(t, name)


def make_component(
    t: MType, preffix: str, handler: ModuleHandler, init: ModuleInit = defaultModuleInit
) -> Component:
    fact = new_component_factory(t, "-".join([preffix, "comp"]))
    return fact.make_component(
        fact.make_signature(
            "-".join([preffix, "sig"]),
            fact.make_module("-".join([preffix, "mod"]), handler, init),
        )
    )

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

    def get_modules(
        self, t: MType | None = None, modules: list[str] = ["*"]
    ) -> list[Module]:
        found = self.find(modules)
        found = [m for _, m in found if m.m_type == t]

        # Deduplicate
        uniq = {id(m): m for m in found}
        return list(uniq.values())

    def build(
        self, types: list[MType] = MFACTORY.all(), modules: list[str] = ["*"]
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

        logger.info(
            "\033[1mRegistry information table.\033[0m  Includes matching available modules."
        )
        if modules != ["*"]:
            logger.info(f"Queries: {', '.join(modules)}")
        logger.info(
            tabulate(
                rows,
                headers=["Collection", "Type", "Module"],
                tablefmt="rounded_outline",
            )
        )


def new_component_manager(study: str) -> ComponentManager:
    return ComponentManager(study)
