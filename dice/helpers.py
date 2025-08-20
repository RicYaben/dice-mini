import pandas as pd

from uuid import uuid4
from dataclasses import dataclass

from dice.models import Source, Label, Model, Fingerprint, FingerprintLabel
from dice.loaders import Loader, jsonl_loader

def new_source(name: str, fpath: str | list[str], id: str | None = None, loader: Loader = jsonl_loader) -> Source:
    p: list[str] = []
    match fpath:
        case str():
            p=[fpath]
        case list():
            p=fpath
    
    if id is None:
        id = str(uuid4())

    return Source(
        id=id,
        name=name,
        paths=p,
        _handler=loader
    ) 

def new_label(module_name: str, name: str, description: str ="-", mitigaton: str ="-") -> Label:
    return Label(
        id=f"{module_name}_{name}",
        name=name,
        description=description,
        mitigation=mitigaton,
        module_name=module_name
    )

def new_fingerprint(module: str, host: str, record: str, data: str, protocol: str = "-", port: int = -1) -> Fingerprint:
    return Fingerprint(
        id=f"{module}_{str(uuid4())}",
        host=host,
        record_id=record,
        module_name=module,
        data=data,
        port=port,
        protocol=protocol
    )

def new_fp_label(fingerprint: str, label: str) -> FingerprintLabel:
    return FingerprintLabel(fingerprint, label)

@dataclass
class Collection:
    items: list[Model]
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([t.to_dict() for t in self.items])
    def add_item(self, item: Model):
        self.items.append(item)

def new_collection(*items: Model) -> Collection:
    return Collection(list(items))

def load_sources(*fpaths: str, name: str="-") -> list[Source]:
    return [new_source(name, p) for p in fpaths]