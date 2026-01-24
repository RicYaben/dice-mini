import pandas as pd
import ipaddress

from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional

from dice.models import Source, Label, Model, Fingerprint, FingerprintLabel, Tag, HostTag, Host
from dice.loaders import Loader, file_loader

def new_source(name: str, fpath: str, loader: Loader = file_loader, batch_size: int = 10_000) -> Source:
    return Source(
        name=name,
        path=fpath,
        batch_size=batch_size,
        _handler=loader
    ) 

def new_label(module_name: str, name: str, short:  Optional[str]= None, description:  Optional[str]= None, mitigaton:  Optional[str]= None) -> Label:
    return Label(
        name=name,
        short=short,
        description=description,
        mitigation=mitigaton,
        module_name=module_name
    )

def new_fingerprint(module: str, host_id: str, record_id: str, data: str, protocol: Optional[str]= None, port: Optional[int]= None) -> Fingerprint:
    return Fingerprint(
        host_id=host_id,
        record_id=record_id,
        module_name=module,
        data=data,
        port=port,
        protocol=protocol
    )

def new_fp_label(fp_id: str, label_id: str) -> FingerprintLabel:
    return FingerprintLabel(fingerprint_id=fp_id, label_id=label_id)

def new_tag(module_name: str, name: str, description: str="-") -> Tag:
    return Tag(
        name=name,
        description=description,
        module_name=module_name
    )

def new_host_tag(host_id: str, tag_id: str, details: Optional[str] = None, protocol: Optional[str]= None, port: Optional[int] = None) -> HostTag:
    return HostTag(host_id=host_id, tag_id=tag_id, details=details, protocol=protocol, port=port)

def new_host(ip: str, domain: str = "", prefix: str ="", asn: str = "") -> Host:
    ipaddress.ip_address(ip) # this panics if not an ip address
    return Host(
        ip=ip,
        domain=domain, 
        prefix=prefix, 
        asn=asn
    )

@dataclass
class Collection:
    items: list[Model]
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([t.to_dict() for t in self.items])
    def add_item(self, item: Model):
        self.items.append(item)

def new_collection(*items: Model) -> Collection:
    return Collection(list(items))

def get_record_field(r, field: str, default: Any=None, prefix: str="data_") -> Any:
    v = r.get(prefix+field, default)

    if isinstance(v, (list, tuple)):
        return default if len(v) == 0 else v
    
    return v if not pd.isna(v) else default

def record_to_dict(r, prefix: str="data_") -> dict:
    d = r.to_dict()
    d = {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
    return d

def with_records(records: Iterable[dict], chunk_size: int = 5_000) -> Loader:
    def load(*args, **kwargs) -> Generator[pd.DataFrame, None, None]:
        batch = []
        for rec in records:
            batch.append(rec)
            if len(batch) >= chunk_size:
                yield pd.DataFrame(batch)
                batch.clear()

        # Yield remaining records
        if batch:
            yield pd.DataFrame(batch)
    return load

def with_model(models: Iterable[Model], chunk_size: int= 5_000) -> Loader:
    def load(*args, **kwargs) -> Generator[pd.DataFrame, None, None]:
        batch = []
        for rec in models:
            batch.append(rec)
            if len(batch) >= chunk_size:
                col = new_collection(*batch)
                yield col.to_df()
                batch.clear()

        # Yield remaining records
        if batch:
            yield pd.DataFrame(batch)
    return load