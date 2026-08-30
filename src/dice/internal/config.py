import logging

from tomlkit import document, dumps, parse, table

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: I really despise globals, this should go away into the configuration
DATA_PREFIX: str = "data_"
DEFAULT_BSIZE: int = 50_000


class Configuration:
    def __init__(self, fpath: str | None) -> None:
        if not fpath:
            self.data = document()
            return
        self.load(fpath)

    def load(self, fpath: str) -> "Configuration":
        logger.info(f"Loading configuration from {fpath}")
        with open(fpath, "+r", encoding="utf-8") as f:
            self.data = parse(f.read())
        return self

    def update(self, module: str, **kwargs) -> None:
        if module not in self.data:
            self.data[module] = table()
        self.data[module].update(**kwargs)

    def update_all(self, **kwargs) -> "Configuration":
        for module, values in kwargs.items():
            self.update(module, **values)
        return self

    def dump(self, fpath: str) -> None:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(dumps(self.data))

    def to_dict(self) -> dict:
        return self.data.unwrap()


def load_configuration(fpath: str | None) -> Configuration:
    return Configuration(fpath)
