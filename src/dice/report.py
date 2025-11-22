from dataclasses import dataclass

@dataclass
class Service:
    port: int
    name: str
    details: dict
    labels: list[str]

@dataclass
class Report:
    host: Host
    services: list[Service]
    tags: list[Tag]

    ip: str
    asn: str
    country: str
    city: str
    prefix: str
    resource: str
    ports: list[int]
    services: list[Service]
    tags: list[dict]

def new_report(host, services) -> Report:
    return Report(
        **host.to_dict(),
        services=new_serviceservices
    )