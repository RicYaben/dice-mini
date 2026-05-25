import ipaddress
from typing import Optional

# from dice.loaders import Loader
from dice.shared.models import (
    Fingerprint,
    Host,
    Fingerprint,
    HostTag,
)


def new_fingerprint(
    module: str,
    host: str,
    record_id: int,
    data: str,
    protocol: str
) -> Fingerprint:
    return Fingerprint(
        host=host,
        record_id=record_id,
        module_name=module,
        data=data,
        protocol=protocol
    )


def new_host_tag(
    host: str,
    tag_id: int,
    details: Optional[str] = None,
    protocol: Optional[str] = None,
    port: Optional[int] = None,
) -> HostTag:
    return HostTag(
        host=host, tag_id=tag_id, details=details, protocol=protocol, port=port
    )


def new_host(ip: str, domain: str = "", prefix: str = "", asn: str = "") -> Host:
    ipaddress.ip_address(ip)  # this panics if not an ip address
    return Host(ip=ip, domain=domain, prefix=prefix, asn=asn)