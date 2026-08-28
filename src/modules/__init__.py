from dice.internal.modules import new_registry

from .example import example_classifier

registry = new_registry("core")
registry.register(example_classifier())

__all__ = [
    "registry"
]
