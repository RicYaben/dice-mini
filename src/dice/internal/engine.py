import logging

from .components import Components
from .config import Configuration
from .monitor import Monitor
from .repository import Repository

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, comps: Components) -> None:
        self.components = comps

    def run(
        self,
        repo: Repository,
        config: Configuration,
        monitor: Monitor,
    ) -> Repository:

        logger.info("preparing (d1)")
        self.components.configure(config)

        logger.info("initializing (d2)")
        self.components.initialize(repo, monitor)

        logger.info("shaking vigorously (d3)")
        self.components.handle()

        return repo

def new_engine(comps: Components) -> Engine:
    return Engine(comps)
