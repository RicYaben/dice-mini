from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

type EventHandler = Callable[[Event], None]


class EventType(StrEnum):
    SOURCE = "source"
    LOAD = "load"
    SANITY = "sanity"


@dataclass(frozen=True)
class Event:
    name: EventType
    summary: dict

    def set_summary(self, s: dict):
        object.__setattr__(self, "summary", s)


def new_event(name: EventType, summary: dict | None = None) -> Event:
    if summary is None:
        summary = {}
    return Event(name, summary)
