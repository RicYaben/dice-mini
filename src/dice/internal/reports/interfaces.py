from collections.abc import Mapping
from typing import Any, Protocol


class ServiceComparator(Protocol):
    @property
    def protocol(self) -> str: ...

    def compare(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """
        Compare two services.

        Return None when they are equivalent, otherwise return
        a description of the differences.
        """
        ...
