from dataclasses import dataclass, field
from importlib.metadata import version

from dice.shared.modules import MFACTORY, ModuleType
from modules import registry

from .components import ComponentManager, Components
from .config import Configuration
from .engine import Engine, new_engine
from .modules import load_registry_plugins

# from .monitor import monitor
from .repository import Repository


@dataclass
class Recipe:
    version: str
    name: str

    configuration: Configuration
    components: Components
    # signatures: Signatures
    requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "configuration": self.configuration.to_dict(),
            "components": self.components.to_dict(),
            "requirements": self.requirements,
        }

@dataclass
class Workflow:
    desc: Recipe
    engine: Engine

    def start(self, repo: Repository) -> None:
        # mon = monitor(self.desc.name)
        self.engine.run(repo, self.desc.configuration) # mon)

    def dump(self) -> str:
        return str(self.desc.to_dict())

class WorkflowBuilder:

    def __init__(self) -> None:
        self._desc = Recipe(
            version=version("dice-mini"),
            name="custom",
            configuration=Configuration(None),
            components=Components([]),
        )
        self._cmanager = ComponentManager().register(registry)

    def configure(self, fpath: str | None) -> 'WorkflowBuilder':
        if fpath is None:
            return self

        self._desc.configuration = self._desc.configuration.load(fpath)
        return self

    def params(self, **kwargs) -> 'WorkflowBuilder':
        self._desc.configuration.update_all(**kwargs)
        return self

    def components(self, t: list[ModuleType], mods: list[str]) -> 'WorkflowBuilder':
        c = self._cmanager.build(t, mods)
        self._desc.components.extend(c)
        return self

    def plugins(self, groups: str | list[str] | None) -> 'WorkflowBuilder':
        if groups is None:
            return self

        if isinstance(groups, str):
            groups = [groups]

        for g in groups:
            if regs := load_registry_plugins(g):
                self._desc.requirements.append(g)
                for r in regs:
                    self._cmanager.register(r)
        return self

    def bake(self) -> Workflow:
        return Workflow(
            desc=self._desc,
            engine=new_engine(self._desc.components),
        )

    def descriptor(self, desc: Recipe) -> 'WorkflowBuilder':
        self._desc = desc
        return self

    def unmarshal(self, data: dict) -> Recipe:
        # This should return a RecipeBuilder, but there are a few things that would change

        # conf
        conf = Configuration(None)
        conf.update_all(**data["configuration"])

        # plugins
        self.plugins(data["requirements"])

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
            configuration=conf,
            components=comps,
            requirements=data["requirements"],
        )

def new_builder() -> WorkflowBuilder:
    return WorkflowBuilder()

def prepare(desc: Recipe) -> WorkflowBuilder:
    return new_builder().descriptor(desc)

def unmarshal(data: dict) -> Recipe:
    return new_builder().unmarshal(data)
