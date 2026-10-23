"""Public HTTP route specifications for ML Platform control-plane clients."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class HttpMethod(StrEnum):
    """HTTP methods supported by the public control-plane contract."""

    DELETE = "DELETE"
    GET = "GET"
    PATCH = "PATCH"
    POST = "POST"


@dataclass(frozen=True)
class Route:
    """One public API method and its URI template."""

    method: HttpMethod
    template: str

    def path(self, **parameters: str | int | UUID) -> str:
        """Render this route for the supplied typed path parameters."""
        return self.template.format(**parameters)


class DatasetRoutes:
    """Public dataset resource routes."""

    LIST = Route(HttpMethod.GET, "/datasets")
    CREATE = Route(HttpMethod.POST, "/datasets")
    DETAIL = Route(HttpMethod.GET, "/datasets/{name}/versions/{version}")
    UPDATE = Route(HttpMethod.PATCH, "/datasets/{name}/versions/{version}")
    DELETE = Route(HttpMethod.DELETE, "/datasets/{name}/versions/{version}")
    MATERIALIZE = Route(
        HttpMethod.POST,
        "/datasets/{name}/versions/{version}/partitions",
    )


class FeatureRoutes:
    """Public feature resource routes."""

    LIST = Route(HttpMethod.GET, "/features")
    CREATE = Route(HttpMethod.POST, "/features")
    DETAIL = Route(HttpMethod.GET, "/features/{name}/versions/{version}")
    DELETE = Route(HttpMethod.DELETE, "/features/{name}/versions/{version}")
    MATERIALIZE = Route(
        HttpMethod.POST,
        "/features/{name}/versions/{version}/partitions",
    )


class LabelRoutes:
    """Public label resource routes."""

    LIST = Route(HttpMethod.GET, "/labels")
    CREATE = Route(HttpMethod.POST, "/labels")
    DETAIL = Route(HttpMethod.GET, "/labels/{name}/versions/{version}")
    DELETE = Route(HttpMethod.DELETE, "/labels/{name}/versions/{version}")
    MATERIALIZE = Route(
        HttpMethod.POST,
        "/labels/{name}/versions/{version}/partitions",
    )


class JobRoutes:
    """Public materialization-job routes."""

    LIST = Route(HttpMethod.GET, "/jobs")
    DETAIL = Route(HttpMethod.GET, "/jobs/{job_id}")


class PartitionRoutes:
    """Public worker partition-completion route."""

    COMPLETE = Route(HttpMethod.POST, "/partitions/complete")
