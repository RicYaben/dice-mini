from collections.abc import Callable

from cyclopts import Parameter, Token


@Parameter(n_tokens=1, accepts_keys=False)
def token_converter(delimiter: str) -> Callable:
    def handler(_: type, tokens: Token):
        return [x.strip() for x in tokens[0].value.split(delimiter) if x.strip()]

    return handler
