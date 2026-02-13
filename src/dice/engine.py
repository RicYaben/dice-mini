from dataclasses import dataclass
from tabulate import tabulate

import logging

from dice.config import (
    CLASSIFIER,
    FINGERPRINTER,
    SCANNER,
    TAGGER,
)
from dice.repo import Repository
from dice.components import Component

logger = logging.getLogger(__name__)

@dataclass
class Engine:
    # list of components registered
    components: list[Component]

    def run(
        self,
        repo: Repository,
    ) -> Repository:

        def fcomp(t):
            return lambda c: c.c_type == t

        logger.info("initializing")
        for c in self.components:
            c.init(repo)

        logger.info("shaking vigorously")
        for m in [SCANNER, FINGERPRINTER, CLASSIFIER, TAGGER]:
            if comps := list(filter(fcomp(m), self.components)):
                logger.info(f"rolling {m.name}(s)")
                for c in comps:
                    c.handle()

        return repo

    def info(self) -> None:
        """
        Print engine info showing components, their type, signatures, and associated modules.
        One row per module, merging repeated Component / Type / Signature cells visually.
        """
        rows = []

        # collect rows: one row per module
        for comp in self.components:
            for sig in comp.signatures:
                for mod in sig.modules:
                    rows.append(
                        [comp.name, str(comp.c_type).upper(), sig.name, mod.collection, mod.name]
                    )

        if not rows:
            logger.info("No components found.")
            return

        # sort rows by Component, Type, Signature, Module
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))

        def collapse_repeated(rows):
            if not rows:
                return rows

            # one "last seen" per column
            num_cols = len(rows[0])
            last_seen = [None] * num_cols

            for row in rows:
                for col in range(num_cols):
                    current = row[col]

                    # Only blank out if the current value matches AND all previous columns are empty
                    if current == last_seen[col] and all(row[i] == "" for i in range(col)):
                        row[col] = ""
                    else:
                        last_seen[col] = current

            return rows

        # merge repeated cells visually
        rows = collapse_repeated(rows)

        logger.info(
            "\033[1mEngine information table.\033[0m  Includes loaded modules by components and signatures."
        )
        logger.info(
            tabulate(
                rows,
                headers=["Component", "Type", "Signature", "Collection", "Module"],
                tablefmt="rounded_outline",
            )
        )


def new_engine(*components: Component) -> Engine:
    return Engine(list(components))

