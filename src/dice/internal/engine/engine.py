import logging

from dice.shared.config import ModuleFlags
from dice.shared.repository import Repository

from .components import Components

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, comps: Components) -> None:
        self.components = comps

    def run(
        self,
        repo: Repository,
        flags: ModuleFlags,
    ) -> Repository:

        logger.info("preparing (d1)")
        self.components.flags(flags)

        logger.info("initializing (d2)")
        self.components.initialize(repo)

        logger.info("shaking vigorously (d3)")
        self.components.handle()

        return repo


def engine(comps: Components) -> Engine:
    return Engine(comps)
