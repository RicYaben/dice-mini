from collections.abc import Mapping, Sequence
from typing import Any

from .interfaces import ServiceComparator
from .models import Comparison, Report, ReportComparison


class ReportProtocolNotFoundException(Exception):
    def __init__(self, report: str) -> None:
        self.report = report
        super().__init__(f"No comparator found for report: {report}")


class ReportComparator:
    def __init__(
        self,
        service_comparators: Sequence[ServiceComparator] = (),
    ) -> None:
        self._service_comparators = {
            comparator.protocol: comparator for comparator in service_comparators
        }

    def compare(
        self,
        reports: Sequence[Report | None],
    ) -> Comparison:
        if not reports:
            raise ValueError("At least one report is required")

        present = [report for report in reports if report is not None]

        if not present:
            raise ValueError("At least one report must be present")

        ip = present[0].ip

        if any(report.ip != ip for report in present):
            raise ValueError("All reports must belong to the same IP")

        baseline = present[0]
        comparisons: list[ReportComparison] = []

        baseline_seen = False

        for report in reports:
            if report is None:
                if baseline_seen:
                    comparisons.append(
                        ReportComparison(
                            baseline=baseline,
                            current=None,
                            changes={
                                "status": "removed",
                            },
                        )
                    )

                continue

            if not baseline_seen:
                baseline_seen = True
                continue

            comparisons.append(
                ReportComparison(
                    baseline=baseline,
                    current=report,
                    changes=self._compare(
                        baseline,
                        report,
                    ),
                )
            )

        return Comparison(
            ip=ip,
            reports=tuple(reports),
            comparisons=tuple(comparisons),
        )

    def _compare(
        self,
        baseline: Report,
        current: Report,
    ) -> dict[str, Any]:
        changes: dict[str, Any] = {}

        self._compare_collection(
            changes,
            "ports",
            baseline.ports,
            current.ports,
        )

        self._compare_services(
            changes,
            baseline.services,
            current.services,
        )

        self._compare_collection(
            changes,
            "tags",
            baseline.tags,
            current.tags,
        )

        self._compare_collection(
            changes,
            "labels",
            baseline.labels,
            current.labels,
        )

        return changes

    @staticmethod
    def _compare_collection(
        changes: dict[str, Any],
        name: str,
        baseline: Sequence[Any],
        current: Sequence[Any],
    ) -> None:
        baseline_set = set(baseline)
        current_set = set(current)

        added = current_set - baseline_set
        removed = baseline_set - current_set

        if not added and not removed:
            return

        changes[name] = {
            "added": sorted(added),
            "removed": sorted(removed),
        }

    def _compare_services(
        self,
        changes: dict[str, Any],
        baseline: Sequence[Mapping[str, Any]],
        current: Sequence[Mapping[str, Any]],
    ) -> None:
        baseline_services = self._service_map(baseline)
        current_services = self._service_map(current)

        added = current_services.keys() - baseline_services.keys()
        removed = baseline_services.keys() - current_services.keys()
        common = baseline_services.keys() & current_services.keys()

        service_changes: dict[str, Any] = {}

        for key in sorted(added):
            service_changes[self._service_name(key)] = {
                "status": "added",
                "current": current_services[key],
            }

        for key in sorted(removed):
            service_changes[self._service_name(key)] = {
                "status": "removed",
                "baseline": baseline_services[key],
            }

        for key in sorted(common):
            baseline_service = baseline_services[key]
            current_service = current_services[key]

            protocol = key[0]
            if not protocol:
                # TODO: fix this
                raise ReportProtocolNotFoundException("")

            comparator = self._service_comparators.get(protocol)

            if comparator is None:
                comparison = self._compare_service_fields(
                    baseline_service,
                    current_service,
                )
            else:
                comparison = comparator.compare(
                    baseline_service,
                    current_service,
                )

            if comparison is not None:
                service_changes[self._service_name(key)] = {
                    "status": "changed",
                    "changes": comparison,
                }

        if service_changes:
            changes["services"] = service_changes

    @staticmethod
    def _service_map(
        services: Sequence[Mapping[str, Any]],
    ) -> dict[
        tuple[str | None, int | None],
        Mapping[str, Any],
    ]:
        result: dict[
            tuple[str | None, int | None],
            Mapping[str, Any],
        ] = {}

        for service in services:
            protocol = service.get("protocol")
            port = service.get("port")

            key = (
                protocol if isinstance(protocol, str) else None,
                port if isinstance(port, int) and not isinstance(port, bool) else None,
            )

            result[key] = service

        return result

    @staticmethod
    def _service_name(
        key: tuple[str | None, int | None],
    ) -> str:
        protocol, port = key

        if port is None:
            return str(protocol)

        return f"{protocol}:{port}"

    @staticmethod
    def _compare_service_fields(
        baseline: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        keys = baseline.keys() | current.keys()

        changes: dict[str, Any] = {}

        for key in keys:
            left = baseline.get(key)
            right = current.get(key)

            if left != right:
                changes[key] = {
                    "baseline": left,
                    "current": right,
                }

        return changes or None
