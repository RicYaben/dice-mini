from dataclasses import dataclass
from typing import Optional, Callable

type EventHandler = Callable[[Event], None]

@dataclass(frozen=True)
class Event:
    name: str
    summary: dict

    def set_summary(self, s: dict):
        object.__setattr__(self, 'summary', s)

def new_event(name: str, summary: dict = {}) -> Event:
    return Event(name, summary)