import ipaddress

from dice.shared.models import (
    Fingerprint,
    Host,
    HostTag,
)


def new_fingerprint(
    module: str, host: str, record_id: int, data: dict, protocol: str
) -> Fingerprint:
    return Fingerprint(
        host=host, record_id=record_id, module_name=module, data=data, protocol=protocol
    )


def new_host_tag(
    host: str,
    tag_id: int,
    details: str | None = None,
    protocol: str | None = None,
    port: int | None = None,
) -> HostTag:
    return HostTag(
        host=host, tag_id=tag_id, details=details, protocol=protocol, port=port
    )


def new_host(ip: str, domain: str = "", prefix: str = "", asn: str = "") -> Host:
    ipaddress.ip_address(ip)  # this panics if not an ip address
    return Host(ip=ip, domain=domain, prefix=prefix, asn=asn)
