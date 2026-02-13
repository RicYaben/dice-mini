from datetime import datetime, timezone
from cryptography import x509
from cryptography.hazmat.primitives import serialization

import pandas as pd

from dice.engine import Module
from dice.query import query_db

def certs_cls_init(mod: Module) -> None:
    mod.register_label(
        "reused-certificate",
        "certificate reused accross multiple addresses",
    )
    mod.register_label(
        "expired-certificate",
        "certificate is expired",
    )
    mod.register_label(
        "long-lasting-certificate",
        "certificate validity period longer than reccommendation",
    )
    mod.register_label(
        "weak-crypto",
        "weak crytographic algorithms",
    )
    mod.register_label(
        "reused-keys",
        "certificate is expired",
    )
    mod.register_label(
        "malformed",
        "certificate is malformed",
    )

    # TODO: check the mining p's and q's paper

def parse_certificate(certificate: str) -> dict | None:
    "parses a X509 certificate"

    header =  "-----BEGIN CERTIFICATE-----"
    if not certificate.startswith(header):
        certificate = "%s\n%s" % (header, certificate)

    footer = "-----END CERTIFICATE-----"
    if (not certificate.endswith(footer)) or (not certificate.endswith(f"{footer}\n")):
        certificate = "%s\n%s\n" % (certificate, footer)

    b = certificate.encode()
    try:
        cert = x509.load_pem_x509_certificate(b)
        pkey = cert.public_key()
    except Exception:
        return

    not_before = cert.not_valid_before_utc.replace(tzinfo=timezone.utc)
    not_after = cert.not_valid_after_utc.replace(tzinfo=timezone.utc) 

    raw_key = pkey.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return dict(
        raw=certificate,
        subject=cert.subject,
        issuer=cert.issuer,
        pkey=pkey,
        raw_key=raw_key,
        signature_algorithm=cert.signature_algorithm_oid._name,
        not_before=not_before,
        not_after=not_after,
    )


def eval_times(mod: Module, fids: str, cert: dict) -> None:
    not_before: datetime = cert.get("not_before") # type: ignore
    not_after: datetime = cert.get("not_after") # type: ignore
    timestamp = datetime.now()

    if not_before > timestamp:
        for fid in fids:
            mod.store(mod.make_label(fid, "malformed", "future"))
        return
    
    if (not_after - not_before) <= pd.Timedelta(days=0):
        for fid in fids:
            mod.store(mod.make_label(fid, "malformed", "negative time"))
        return
    
    if not_after < timestamp:
        for fid in fids:
            mod.store(mod.make_label(fid, "expired-certificate"))

    if (not_after - not_before) > pd.Timedelta(days=5*365.25):
        for fid in fids:
            mod.store(mod.make_label(fid, "long-lasting-certificate"))

def eval_crypto(mod: Module, fids: str, cert: dict) -> None:
    sig: str = cert.get("signature_algorithm", "")
    if not sig: 
        return
    
    separator = "With" if "With" in sig else "-with-"
    hash_func = sig.split(separator)[0]

    # Check hash function
    # SHA-1: deprecated in 2011
    if hash_func in ["md5", "sha1", "dsa"]:
        for fid in fids:
            mod.store(mod.make_label(fid, "weak-crypto"))

def eval_key(mod: Module, fids: str, cert: dict) -> None:
        key = cert.get("pkey")
        if pd.isna(key):
            return

        if key.key_size < 2048:
            for fid in fids:
                mod.store(mod.make_label(fid, "short-key"))

def cert_eval_handler(mod: Module) -> None:
    con = mod.repo().get_connection()
    q = "..."
    for cert_df in mod.query(query_db("certificates")):
        for _, cert in cert_df.iterrows():
            fids = con.execute(q.format(cert=cert)).df()["id"].tolist()
            eval_crypto(mod, fids, cert)
            eval_key(mod, fids, cert)
            eval_times(mod, fids, cert)

def cert_reuses(mod: Module) -> None:
    # get fingerprints with raw certificates found > 1 times
    q_fps_reused_certs = "..."
    for ch in mod.query(q_fps_reused_certs):
        for fid in ch["id"].tolist():
            mod.store(mod.make_label(fid, "reused-certificate"))

    q_fps_reused_keys = "..."
    for ch in mod.query(q_fps_reused_keys):
        for fid in ch["id"].tolist():
            mod.store(mod.make_label(fid, "reused-keys"))

def scan_certificates(mod: Module) -> None:
    def handler(df: pd.DataFrame):
        for _, fp in df.iterrows():
            certs = fp.get("data_certificates")
            if not certs:
                return
            
            cert = parse_certificate(certs[0])
            if not cert:
                return
            
            mod.store(cert)
    mod.with_pbar(handler, query_db("fingerprints", prefix=""), desc="certificates")
        