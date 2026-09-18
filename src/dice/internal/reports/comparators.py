from collections.abc import Mapping
from typing import Any

from .interfaces import ServiceComparator


class IEC104Comparator(ServiceComparator):
    def supports(self, service: Mapping[str, Any]) -> bool:
        return service.get("protocol") == "iec104"

    def compare(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        # IEC-104-specific comparison here.
        ...


class FoxComparator(ServiceComparator):
    def supports(self, service: Mapping[str, Any]) -> bool:
        return service.get("protocol") == "fox"

    def compare(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        # FOX-specific comparison here.
        ...
