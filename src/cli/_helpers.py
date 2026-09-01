import logging

import typer
import ujson

from dice.internal.config import (
    DatabasesConf,
    DiceConfig,
    LogsConf,
    ModuleFlags,
    ModulesConf,
    load_flags,
    make_configuration,
)


def modules_flags(
    registries: str | None = typer.Option(
        None,
        "-r",
        "--registries",
        help="Load module registries",
    ),
    flags: str | None = typer.Option(
        None,
        "-Mf",
        "--module-flags",
        help="Load module flags from a file",
    ),
    params: str | None = typer.Option(
        None,
        "-Mp",
        "--module-params",
        help="Load module params from a file",
    ),
) -> ModulesConf:
    conf = ModulesConf()

    if registries is not None:
        conf.registries = registries.split(",")

    if flags is not None:
        conf.flags = load_flags(flags)

    if params is not None:
        conf.flags.update_all(**ujson.loads(params))

    return conf

def modules_flags_overrides(
    registries: str | None = typer.Option(
        None,
        "-r",
        "--registries",
        help="Load module registries",
    ),
    flags: str | None = typer.Option(
        None,
        "-Mf",
        "--module-flags",
        help="Load module flags from a file",
    ),
    params: str | None = typer.Option(
        None,
        "-Mp",
        "--module-params",
        help="Load module params from a file",
    ),
) -> dict:
    overrides = {}

    if registries is not None:
        overrides["registries"] = registries.split(",")

    if flags is not None:
        overrides["flags"] = load_flags(flags)

    if params is not None:
        overrides.setdefault("flags", ModuleFlags({}))
        overrides["flags"].update_all(**ujson.loads(params))

    return overrides


def logs_flags(
    log_level: str = typer.Option(
        "info", "-Ll", "--log-level", help="Log level"
    ),
    log_file: str | None = typer.Option(
        None, "-Lf", "--log-file", help="Log file"
    ),
    log_error: str | None = typer.Option(
        None, "-Le", "--log-error", help="Log error"
    ),
) -> LogsConf:
    level = getattr(logging, log_level.upper())
    return LogsConf(level=level, file=log_file, errors=log_error)


def database_flags(
    db: str | None = typer.Option(
        None, "-db", "--database", help="Results database"
    ),
    cookbook: str | None = typer.Option(
        None, "-cb", "--cookbook", help="Cookbook"
    ),
) -> DatabasesConf:
    return DatabasesConf(results=db, cookbook=cookbook)

def database_flags_override(
    db: str | None = typer.Option(
        None, "-db", "--database", help="Results database"
    ),
    cookbook: str | None = typer.Option(
        None, "-cb", "--cookbook", help="Cookbook"
    ),
) -> dict:
    return {"results": db, "cookbook": cookbook}


def config_flags(
    configuration: str | None = typer.Option(
        None,
        "-c",
        "--configuration",
        help="Path to configuration file",
    ),
    query_bsize: int | None = typer.Option(
        None,
        "-q",
        "--query-bsize",
        help="Query batch size",
    ),
    health: str | None = typer.Option(
        None,
        "-h",
        "--health",
        help="Health endpoints",
    ),
) -> DiceConfig:
    conf = make_configuration(configuration)

    if query_bsize is not None:
        conf.query_bsize = query_bsize

    if health is not None:
        conf.health = health.split(",")

    return conf

def config_group_flags(
    config: DiceConfig = typer.Depends(config_flags)
) -> dict:
    return {}
