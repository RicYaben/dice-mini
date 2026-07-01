from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Flag:
    value: Any
    description: str = ""


def flag(default: Any, description: str = "") -> Any:
    return Flag(default, description)


class FlagsMeta(type):
    def __new__(mcls, name, bases, ns):
        annotations = ns.get("__annotations__", {})

        meta = {}

        for k, _ in annotations.items():
            val = ns.get(k)

            if isinstance(val, Flag):
                meta[k] = val
                ns[k] = val.value
            else:
                meta[k] = Flag(value=val)
                ns[k] = val

        ns["__meta__"] = meta
        return super().__new__(mcls, name, bases, ns)


class Flags(metaclass=FlagsMeta):
    __meta__: dict[str, Flag]

    def __init__(self, **overrides):
        cls = self.__class__

        meta = getattr(cls, "__meta__", {})

        for name, m in meta.items():
            value = overrides.get(name, m.value)
            setattr(self, name, value)

        self.__meta__ = meta

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown flag: {k}")
            setattr(self, k, v)
