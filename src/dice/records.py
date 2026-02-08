from dataclasses import dataclass
from enum import StrEnum
from typing import Optional
from dacite import from_dict

import pandas as pd

class Connection(StrEnum):
    CONNECTED = "connected"
    REFUSED = "refused"

class Authentication(StrEnum):
    UNAUTHORIZED = "unauthorized"
    ANONYMOUS = "anonymous"
    SELF_SIGNED_CERTIFICATE = "self-signed-certificate"

class Access(StrEnum):
    WRITE = "write"
    READ = "read"
    EXECUTE = "execute"

class Maturity(StrEnum):
    ACTIVE = "active"
    MATURE = "mature"
    DEPRECATED = "deprecated"

@dataclass
class Service:
    name: str
    version: str
    cpe: Optional[str]

@dataclass
class Record:
    # conn
    connection: Connection
    # encryption
    encryption: Optional[str]
    certificates: list[str] 
    # authentication & authorization
    authentication: Authentication
    access: list[Access]
    # service details
    service: Service
    

def make_record(row: pd.Series) -> Record:
    data = row.to_dict()
    return from_dict(Record, data)


def eval_communication(r: pd.Series) -> bool:
    "determine whether the communication followed the protocol"
    ...


def eval_status(r: pd.Series) -> Connection:
    "check if the connection was refused, allowed to connect, or not even the right com"
    return Connection.CONNECTED

def eval_encryption(r: pd.Series) -> str | None:
    "check the encryption used"
    scheme = r.get("data_scheme")
    match scheme:
        case "ssl", "tls":
            return "TLS"
        case "dtls":
            return "DTLS"
        