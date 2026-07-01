import logging

from tabulate import tabulate

from dice.shared.modules import MFACTORY

from .components import Component
from .repository import Repository

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, comps: list[Component]) -> None:
        self.components = comps

    def run(
        self,
        repo: Repository,
    ) -> Repository:

        logger.info("initializing (d1)")
        for c in self.components:
            c.initialize(repo)

        logger.info("shaking vigorously (d2)")
        rcomps = self.components
        for m in MFACTORY.all():
            if comps := list(filter(lambda x: x.t == m, rcomps)):
                logger.info(f"rolling {m.name}(s)")
                for c in comps:
                    c.handle()
                    rcomps.remove(c)

        return repo

    def info(self) -> None:
        """
        Print engine info showing components, their type, signatures, and associated modules.
        One row per module, merging repeated Component / Type / Signature cells visually.
        """
        rows = []

        # collect rows: one row per module
        # TODO: add the registry where the module is located?
        for comp in self.components:
            for sig in comp.signatures:
                for mod in sig.modules:
                    rows.append(
                        [
                            comp.name,
                            str(comp.t).upper(),
                            sig.name,
                            mod.desc.name,
                        ]
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
                    if current == last_seen[col] and all(
                        row[i] == "" for i in range(col)
                    ):
                        row[col] = ""
                    else:
                        last_seen[col] = current

            return rows

        # merge repeated cells visually
        rows = collapse_repeated(rows)

        print(
            "\033[1mEngine information table.\033[0m  Includes loaded modules by components and signatures."
        )
        print(
            tabulate(
                rows,
                headers=["Component", "Type", "Signature", "Module"],
                tablefmt="rounded_outline",
            )
        )


def new_engine(*components: Component) -> Engine:
    return Engine(list(components))
