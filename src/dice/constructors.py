import ipaddress
import pandas as pd

from dataclasses import dataclass
from typing import Generator, Iterable, Optional

from dice.loaders import Loader
from dice.models import Fingerprint, FingerprintLabel, Host, Model, Label, Fingerprint, Tag, HostTag, Source

def new_source(name: str) -> Source:
    return Source(
        name=name,
    ) 

def new_label(module_name: str, name: str, short:  Optional[str]= None, description:  Optional[str]= None, mitigaton:  Optional[str]= None, level: int=0) -> Label:
    return Label(
        name=name,
        short=short,
        description=description,
        mitigation=mitigaton,
        module_name=module_name,
        level=level,
    )

def new_fingerprint(module: str, host: str, record_id: int, resource_id: int, data: str, protocol: Optional[str]= None, port: Optional[int]= None) -> Fingerprint:
    return Fingerprint(
        host=host,
        record_id=record_id,
        resource_id=resource_id,
        module_name=module,
        data=data,
        port=port,
        protocol=protocol
    )

def new_fp_label(fp_id: int, label_id: int) -> FingerprintLabel:
    return FingerprintLabel(fingerprint_id=fp_id, label_id=label_id)

def new_tag(module_name: str, name: str, description: str="-") -> Tag:
    return Tag(
        name=name,
        description=description,
        module_name=module_name
    )

def new_host_tag(host: str, tag_id: int, details: Optional[str] = None, protocol: Optional[str]= None, port: Optional[int] = None) -> HostTag:
    return HostTag(host=host, tag_id=tag_id, details=details, protocol=protocol, port=port)

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


def with_items(*items: Model) -> pd.DataFrame:
    return new_collection(*items).to_df()