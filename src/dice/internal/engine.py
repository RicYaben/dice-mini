import logging

from .components import Components
from .config import ModuleFlags

# from .monitor import Monitor
from .repository import Repository

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, comps: Components) -> None:
        self.components = comps

    def run(
        self,
        repo: Repository,
        flags: ModuleFlags,
        # monitor: Monitor,
    ) -> Repository:

        logger.info("preparing (d1)")
        self.components.flags(flags)

        logger.info("initializing (d2)")
        self.components.initialize(repo)  # , monitor)

        logger.info("shaking vigorously (d3)")
        self.components.handle()

        return repo


def new_engine(comps: Components) -> Engine:
    return Engine(comps)
