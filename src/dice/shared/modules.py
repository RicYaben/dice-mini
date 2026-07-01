import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Optional, Protocol, TypeVar, cast

from .flags import Flags
from .models import Label, Tag
from .repository import BaseRepo


@dataclass(frozen=True)
class ModuleType:
    command: str
    name: str
    alias: str

    def alias_name(self) -> str:
        return self.alias or self.name

    def __str__(self) -> str:
        return self.name


class ModuleFactory:
    def __init__(self):
        self._lookup: OrderedDict[str, ModuleType] = OrderedDict()

    def register(self, mt: ModuleType):
        # Map each of the three identifiers to the same MType
        for key in (mt.command, mt.name, mt.alias):
            if key:  # allows alias=None
                self._lookup[key] = mt

    def get(self, key: str) -> ModuleType:
        try:
            return self._lookup[key]
        except KeyError:
            raise KeyError(f"No module type found for key: {key!r}")

    def all(self) -> list[ModuleType]:
        # dedupe by .command
        ret = []
        for v in self._lookup.values():
            if v not in ret:
                ret.append(v)
        return ret


class ModuleEnum(Enum):
    SCANNER = ModuleType(command="scan", name="scanner", alias="s")
    CLASSIFIER = ModuleType(command="classify", name="classifier", alias="c")
    FINGERPRINTER = ModuleType(command="fingerprint", name="fingerprinter", alias="f")
    TAGGER = ModuleType(command="tag", name="tag", alias="t")


MFACTORY = ModuleFactory()
for m in ModuleEnum:
    MFACTORY.register(m.value)


def find_module(mod: str) -> ModuleType:
    return MFACTORY.get(mod)


R = TypeVar("R", bound=BaseRepo)
F = TypeVar("F", bound=Flags)


class Runner(Protocol[R, F]):
    def __call__(
        self,
        repo: R,
        flags: F,
        logger: logging.Logger,
    ) -> None: ...


def do_nothing(repo: BaseRepo, flags: Flags, logger: logging.Logger) -> None:
    return


@dataclass
class ModuleDescriptor(Generic[R, F]):
    t: str
    name: str
    description: Optional[str] = None
    flags: type[F] = field(default=cast(type[F], Flags))
    labels: list["Label"] = field(default_factory=list)
    tags: list["Tag"] = field(default_factory=list)

    run_fn: Runner[R, F] = field(default=do_nothing, repr=False)
    pre_fn: Runner[R, F] = field(default=do_nothing, repr=False)
    post_fn: Runner[R, F] = field(default=do_nothing, repr=False)

    _repo: Optional[R] = field(default=None, init=False, repr=False)
    _logger: Optional[logging.Logger] = field(default=None, init=False, repr=False)
    _flags: Optional[F] = field(default=None, init=False, repr=False)

    def initialize(self, repo: BaseRepo, logger: logging.Logger) -> None:
        self.repo = cast(R, repo)
        self.logger = logger
        self.rflags = self.flags

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            raise Exception("module not initialized")
        return self._logger

    @logger.setter
    def logger(self, l: logging.Logger) -> None:
        self._logger = l

    @property
    def repo(self) -> R:
        if not self._repo:
            raise Exception("module not initialized")
        return self._repo

    @property
    def rflags(self) -> F:
        if not self._flags:
            raise Exception("module not initialized")
        return self._flags

    @rflags.setter
    def rflags(self, f: type[F]) -> None:
        self._flags = f()

    @repo.setter
    def repo(self, r: R) -> None:
        self._repo = r

    def run(self) -> None:
        self.run_fn(self.repo, self.rflags, self.logger)
        self.repo.flush()

    def pre(self) -> None:
        self.pre_fn(self.repo, self.rflags, self.logger)
        self.repo.flush()

    def post(self) -> None:
        self.post_fn(self.repo, self.rflags, self.logger)
        self.repo.flush()

    def __str__(self) -> str:
        lines: list[str] = []
        m = find_module(self.t)
        lines.append(f"module: {self.name} ({m.name})")

        if self.description:
            lines.append(f"description: {self.description}")

        meta = getattr(self.flags, "__meta__", {})
        if meta:
            lines.append("flags:")

            for name, f in meta.items():
                lines.append(f"--{name}\t{f.value}\t{f.description}")

        # TODO: labels and tags would go nice into a table
        if self.labels:
            lines.append("labels:")

            for l in self.labels:
                desc = getattr(l, "description", "")
                lines.append(f"-{l.name}\t{desc}")

        if self.tags:
            lines.append("tags:")

            for t in self.tags:
                desc = getattr(t, "description", "")
                lines.append(f"-{t.name}\t{desc}")

        return "\n".join(lines)


# TODO: this is pagination with a bar
# def zgrab2_handler(
#     mod: Module,
#     fp_cb: FPCallback,
#     protocol: str,
# ) -> RowHandler:

#     def handler(r: pd.Series):
#         # TODO: this in the future
#         # is_proto = eval_communication(r), # true or false

#         # # return early, is a false-positive
#         # if not is_proto:
#         #     return

#         # base = {
#         #     "is_protocol": is_proto,
#         #     "connection": eval_status(r), # connected, refused
#         #     "encryption": eval_encryption(r), # TLS, DTLS, or whatever other scheme; otherwise None
#         #     "certificates": r.get("data_certificates", None)
#         # }

#         if fp := fp_cb(r):
#             #base.update(fp)
#             mod.store(mod.make_fingerprint(r, fp, protocol))
#     return handler

# def make_fp_handler(
#     fp_cb: FPCallback,
#     protocol: str = "-",
#     source: str = "zgrab2",
# ) -> ModuleHandler:
#     def wrapper(mod: Module) -> None:
#         match source:
#             case "zgrab2":
#                 h = zgrab2_handler(mod, fp_cb, protocol)
#             case _:
#                 h = default_handler(mod, fp_cb, protocol)

#         q = query_records(source=source, protocol=protocol)
#         norm = get_normalizer(source)
#         mod.itemize(q, h, orient="rows", norm=norm)
#     return wrapper

# def make_cls_handler(
#     cls_cb: Callable[[pd.Series], str | None], protocol="-"
# ) -> ModuleHandler:
#     def wrapper(mod: Module) -> None:
#         repo = mod.repo()

#         def handler(df: pd.DataFrame):
#             labs = []
#             for _, fp in df.iterrows():
#                 if lab := cls_cb(fp):
#                     labs.append(mod.make_label(fp["id"].hex, lab))

#             if not labs:
#                 return

#             with repo.session() as ses:
#                 insert_or_ignore(ses, FingerprintLabel, labs)

#         q = query("fingerprint", protocol=protocol)
#         mod.with_pbar(handler, q)

#     return wrapper
