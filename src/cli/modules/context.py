from dataclasses import dataclass

from dice.cli.config.args import ModulesArg, RegistriesArg


@dataclass
class Context:
    modules: ModulesArg | None = None
    registries: RegistriesArg | None = None
