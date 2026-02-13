import pandas as pd
import ipaddress

from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, Optional

from dice.components import Component, make_component
from dice.config import CLASSIFIER, FINGERPRINTER
from dice.database import insert_or_ignore
from dice.models import Source, Label, Model, Fingerprint, FingerprintLabel, Tag, HostTag, Host
from dice.loaders import Loader
from dice.modules import Module, ModuleHandler, ModuleInit, defaultModuleInit
from dice.query import query_db, query_records

from dice.records import eval_communication, eval_encryption, eval_status

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

def new_fingerprinter(
    handler: ModuleHandler, init: ModuleInit = defaultModuleInit, preffix: str = "fp"
) -> Component:
    return make_component(FINGERPRINTER, preffix, handler, init)


def new_classifier(
    handler: ModuleHandler, init: ModuleInit = defaultModuleInit, preffix: str = "cls"
) -> Component:
    return make_component(CLASSIFIER, preffix, handler, init)


FPCallback = Callable[[pd.Series], dict | None]
RowHandler = Callable[[pd.Series], None]

def default_handler(
    mod: Module,
    fp_cb: FPCallback,
    protocol: str,
) -> RowHandler:
    def handler(r: pd.Series):
        if fp := fp_cb(r):
            mod.store(mod.make_fingerprint(r, fp, protocol))
    return handler

def zgrab2_handler(
    mod: Module,
    fp_cb: FPCallback,
    protocol: str,
) -> RowHandler:
    def handler(r: pd.Series):
        is_proto = eval_communication(r), # true or false

        # return early, is a false-positive
        if not is_proto:
            return

        base = {
            "is_protocol": is_proto,
            "connection": eval_status(r), # connected, refused
            "encryption": eval_encryption(r), # TLS, DTLS, or whatever other scheme; otherwise None
            "certificates": r.get("data_certificates", None)
        }

        if fp := fp_cb(r):
            base.update(fp)

        mod.store(mod.make_fingerprint(r, base, protocol))
    return handler

def make_fp_handler(
    fp_cb: FPCallback,
    protocol: str = "-",
    source: str = "zgrab2",
) -> ModuleHandler:
    def wrapper(mod: Module) -> None:
        match source:
            case "zgrab2":
                h = zgrab2_handler(mod, fp_cb, protocol)
            case _:
                h = default_handler(mod, fp_cb, protocol)

        q = query_records(source=source, protocol=protocol)
        mod.itemize(q, h, orient="rows")
    return wrapper

def make_cls_handler(
    cls_cb: Callable[[pd.Series], str | None], protocol="-"
) -> ModuleHandler:
    def wrapper(mod: Module) -> None:
        repo = mod.repo()

        def handler(df: pd.DataFrame):
            labs = []
            for _, fp in df.iterrows():
                if lab := cls_cb(fp):
                    labs.append(mod.make_label(fp["id"].hex, lab))

            if not labs:
                return

            with repo.session() as ses:
                insert_or_ignore(ses, FingerprintLabel, labs)

        q = query_db("fingerprints", protocol=protocol)
        mod.with_pbar(handler, q)

    return wrapper

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