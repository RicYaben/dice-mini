from dataclasses import dataclass

from dice.internal.components import ComponentManager
from dice.internal.config import Configuration
from dice.internal.engine import Engine, new_engine
from dice.internal.modules import load_registry_plugins
from dice.internal.repository import Repository
from dice.shared.modules import ModuleType
from modules import registry


@dataclass
class Descriptor:
    id: str
    name: str
    modules: list[str]

    remote: str | None = None
    local: str | None = None

    params: dict | None = None
    config: str | None = None
    plugins: str | None = None

@dataclass
class Recipe:
    desc: Descriptor
    config: Configuration
    engine: Engine

    def start(self, repo: Repository) -> Result:
        mon = monitor(self.desc.id, self.desc.name)
        res = self.engine.run(repo, self.config, mon)
        return res

    def dump(self) -> str:
        return str(self.desc)

class RecipeBuilder:

    def __init__(self) -> None:
        self._desc = Descriptor(
            id="custom",
            name="custom",
            modules=[],
        )
        self._config = Configuration(None)
        self._cmanager = ComponentManager().register(registry)

    def configure(self, fpath: str | None) -> 'RecipeBuilder':
        if fpath is None:
            return self

        self._config = self._config.load(fpath)
        return self

    def params(self, **kwargs) -> 'RecipeBuilder':
        self._config.update_all(**kwargs)
        return self

    def engine(self, t: list[ModuleType], mods: list[str]) -> 'RecipeBuilder':
        c = self._cmanager.build(t, mods)
        self._desc.modules = [mod.desc.name for mod in c.modules()]
        self._engine = new_engine(c)
        return self

    def plugins(self, group: str | None) -> 'RecipeBuilder':
        if group is None:
            return self
        for r in load_registry_plugins(group):
            self._cmanager.register(r)
        return self

    def bake(self) -> Recipe:
        return Recipe(
            desc=self._desc,
            config=self._config,
            engine=self._engine,
        )

    def descriptor(self, desc: Descriptor) -> 'RecipeBuilder':
        self._desc = desc

        if desc.config:
            self.configure(desc.config)
        if desc.params:
            self.params(**desc.params)

        t, mods = parse_modules(desc.modules)
        self.engine(t, mods)
        self.plugins(desc.plugins)
        return self

def new_builder() -> RecipeBuilder:
    return RecipeBuilder()

def prepare(desc: Descriptor) -> RecipeBuilder:
    return new_builder().descriptor(desc)
