import logging
from collections.abc import Callable

from .events import Event, EventType, new_event

logger = logging.getLogger(__name__)

type HealthCheck = Callable[[Event], None]


class HealthMonitor:
    def __init__(
        self,
        on_init: list[HealthCheck],
        on_synchronize: list[HealthCheck],
    ) -> None:
        self._on_init = on_init
        self._on_synchronize = on_synchronize

    def initialize(self):
        self.load(new_event(EventType.LOAD))

    def load(self, e: Event):
        logger.info("running initialization checks")
        for check in self._on_init:
            check(e)

    def synchronize(self, e: Event):
        logger.info("running synchronization checks")
        for check in self._on_synchronize:
            check(e)

    def sanity_check(self):
        logger.info("checking repo health")
        e = new_event(EventType.SANITY)
        self.synchronize(e)


def new_health_monitor(
    init: list[HealthCheck] | None = None, sync: list[HealthCheck] | None = None
) -> HealthMonitor:
    if init is None:
        init = []
    if sync is None:
        sync = []
    return HealthMonitor(init, sync)
