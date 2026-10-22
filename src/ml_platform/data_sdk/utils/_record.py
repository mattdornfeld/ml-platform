"""Immutable record decorator for transform contracts."""

from abc import update_abstractmethods
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Signature, signature
from typing import dataclass_transform

from ml_platform.data_sdk._exceptions import DataSdkError


class RecordIdentityError(DataSdkError):
    """Raised when the record decorator cannot preserve decorated class identity."""


class RecordInitializationError(DataSdkError):
    """Raised when a record receives both public and private names for one field."""


@dataclass_transform(frozen_default=True)
def record[Record](record_type: type[Record]) -> type[Record]:
    """Create a frozen dataclass that fulfills inherited property contracts.

    ``dataclass_transform`` exposes the dataclass constructor and the annotated
    abstract-property implementation to mypy and PyCharm. At runtime, an
    annotated property such as ``config`` is backed by ``_config`` and exposed
    through a read-only property. Unlike ``dataclassabc``, this decorator mutates
    and returns the supplied class, preserving identity for ``isinstance`` and
    structural ``match`` statements.
    """
    annotations = dict(getattr(record_type, "__annotations__", {}))
    inherited_properties = {
        name
        for base in record_type.__mro__[1:]
        for name in vars(base)
        if isinstance(getattr(base, name, None), property)
    }
    declared_properties = tuple(
        name for name in annotations if name in inherited_properties
    )
    inherited_record_properties = {
        field_name.removeprefix("_")
        for base in record_type.__mro__[1:]
        for field_name in getattr(base, "__dataclass_fields__", {})
        if field_name.startswith("_")
        and isinstance(
            getattr(base, field_name.removeprefix("_"), None), property
        )
    }
    property_fields = tuple(
        dict.fromkeys((*declared_properties, *inherited_record_properties))
    )
    for name in declared_properties:
        annotations[f"_{name}"] = annotations.pop(name)
    record_type.__annotations__ = annotations

    frozen_record = dataclass(frozen=True)(record_type)
    if frozen_record is not record_type:
        raise RecordIdentityError(
            f"record must preserve the decorated class identity for {record_type.__name__}."
        )
    generated_init = frozen_record.__init__

    def public_init(self: object, *args: object, **kwargs: object) -> None:
        """Accept public property names while storing their private backing fields."""
        for name in property_fields:
            private_name = f"_{name}"
            if name in kwargs:
                if private_name in kwargs:
                    raise RecordInitializationError(
                        f"Record {record_type.__name__} received both {name!r} and {private_name!r}."
                    )
                kwargs[private_name] = kwargs.pop(name)
        generated_init(self, *args, **kwargs)

    private_names = {f"_{name}" for name in property_fields}
    generated_signature = signature(generated_init)
    public_parameters = [
        parameter.replace(name=parameter.name.removeprefix("_"))
        if parameter.name in private_names
        else parameter
        for parameter in generated_signature.parameters.values()
    ]
    generated_annotations = getattr(generated_init, "__annotations__", {})
    public_annotations = {
        name.removeprefix("_") if name in private_names else name: annotation
        for name, annotation in generated_annotations.items()
    }
    public_init.__dict__["__signature__"] = Signature(
        public_parameters,
        return_annotation=generated_signature.return_annotation,
    )
    public_init.__annotations__ = public_annotations
    type.__setattr__(frozen_record, "__init__", public_init)

    for name in declared_properties:
        private_name = f"_{name}"
        setattr(frozen_record, name, property(_property_getter(private_name)))
    update_abstractmethods(frozen_record)
    return frozen_record


def _property_getter(private_name: str) -> Callable[[object], object]:
    """Build the read-only getter used for a record's property-backed field."""

    def get_property(instance: object) -> object:
        """Read the immutable value stored in the private backing field."""
        return getattr(instance, private_name)

    return get_property
