from dataclasses import dataclass, field
from importlib.metadata import version

from dice.internal.engine import (
    ComponentManager,
    Components,
    Engine,
    engine,
    registries,
)
from dice.shared.config import ModuleFlags
from dice.shared.modules import MFACTORY, ModuleType
from dice.shared.repository import Repository
from modules import registry


@dataclass
class Recipe:
    version: str
    name: str

    flags: ModuleFlags
    components: Components
    requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "flags": self.flags.root,
            "components": self.components.to_dict(),
            "requirements": self.requirements,
        }


@dataclass
class Workflow:
    desc: Recipe
    engine: Engine

    def start(self, repo: Repository) -> None:
        self.engine.run(repo, self.desc.flags)  # mon)

    def dump(self) -> str:
        return str(self.desc.to_dict())


class WorkflowBuilder:
    def __init__(self) -> None:
        self._desc = Recipe(
            version=version("dice-mini"),
            name="custom",
            flags=ModuleFlags({}),
            components=Components([]),
        )
        self._cmanager = ComponentManager().register(registry)

    def flags(self, **kwargs) -> "WorkflowBuilder":
        self._desc.flags.update_all(**kwargs)
        return self

    def components(
        self, t: list[ModuleType] | None = None, mods: list[str] | None = None
    ) -> "WorkflowBuilder":
        c = self._cmanager.build(t, mods)
        self._desc.components.extend(c)
        return self

    def registries(self, groups: str | list[str] | None) -> "WorkflowBuilder":
        if groups is None:
            return self

        if isinstance(groups, str):
            groups = [groups]

        for g in groups:
            if regs := registries(g):
                self._desc.requirements.append(g)
                for r in regs:
                    self._cmanager.register(r)
        return self

    def bake(self) -> Workflow:
        return Workflow(
            desc=self._desc,
            engine=engine(self._desc.components),
        )

    def descriptor(self, desc: Recipe) -> "WorkflowBuilder":
        self._desc = desc
        return self

    def unmarshal(self, data: dict) -> Recipe:
        # This should return a RecipeBuilder, but there are a few things that would change

        # conf
        conf = ModuleFlags({})
        conf.update_all(**data["configuration"])

        # plugins
        self.registries(data["requirements"])

        # comps
        comps = Components([])
        for c in data["components"]:
            mods = [":".join([mod["registry"], mod["name"]]) for mod in c["modules"]]
            t = MFACTORY.get(c["type"])
            if comp := self._cmanager.make(t, mods):
                comps.add(comp)

        return Recipe(
            version=data["version"],
            name=data["name"],
            flags=conf,
            components=comps,
            requirements=data["requirements"],
        )


def new_builder() -> WorkflowBuilder:
    return WorkflowBuilder()


def from_recipe(desc: Recipe) -> WorkflowBuilder:
    return new_builder().descriptor(desc)


def unmarshal(data: dict) -> Recipe:
    return new_builder().unmarshal(data)
