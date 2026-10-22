"""Tests for immutable records that implement abstract property contracts."""

import unittest
from abc import ABC, abstractmethod
from dataclasses import FrozenInstanceError
from inspect import signature

from ml_platform.data_sdk.configs._configs import PartitionType
from ml_platform.data_sdk.utils import record


class Partitioned(ABC):
    """Minimal abstract contract used to exercise the record decorator."""

    @property
    @abstractmethod
    def partition_type(self) -> str:
        """Return the output partition cadence."""


class Configured(ABC):
    """Minimal transform-style contract with an abstract configuration property."""

    @property
    @abstractmethod
    def config(self) -> str:
        """Return immutable configuration supplied at construction."""


@record
class PartitionedRecord(Partitioned):
    """Concrete immutable record whose field fulfills the parent property."""

    partition_type: str
    name: str


@record
class TypedPartitionedRecord(Partitioned):
    """Typed variant used to exercise exhaustive enum case matching."""

    partition_type: PartitionType
    name: str


@record
class ConfiguredRecord(Configured):
    """Transform-style record whose public config field fulfills its contract."""

    config: str


class RecordTest(unittest.TestCase):
    """Verify records provide required abstract properties from constructor values."""

    def test_record_preserves_the_original_class_object(self) -> None:
        """The decorator does not replace the type used by class patterns."""

        class RawRecord(Partitioned):
            """An undecorated record used to assert type identity."""

            partition_type: str

        decorated_record = record(RawRecord)

        self.assertIs(decorated_record, RawRecord)
        value = decorated_record(partition_type="daily")
        match value:
            case RawRecord(partition_type="daily"):
                matched = True
            case _:
                matched = False

        self.assertTrue(matched)

    def test_constructor_field_implements_abstract_property(self) -> None:
        """The public constructor accepts the abstract property name."""
        value = PartitionedRecord(partition_type="daily", name="market_data")

        self.assertEqual(value.partition_type, "daily")
        self.assertIn("partition_type", str(signature(PartitionedRecord)))
        with self.assertRaises(FrozenInstanceError):
            value.partition_type = "hourly"  # type: ignore[misc]

    def test_config_field_fulfills_abstract_property_and_supports_class_patterns(
        self,
    ) -> None:
        """A constructor config field becomes an immutable concrete property."""
        value = ConfiguredRecord(config="label-definition")

        self.assertIsInstance(value, ConfiguredRecord)
        self.assertEqual(value.config, "label-definition")
        match value:
            case ConfiguredRecord(config="label-definition"):
                matched = True
            case _:
                matched = False

        self.assertTrue(matched)
        with self.assertRaises(FrozenInstanceError):
            value.config = "replacement"  # type: ignore[misc]

    def test_partition_type_preserves_its_enum_type_for_case_matching(
        self,
    ) -> None:
        """Pattern matching receives the declared PartitionType enum rather than a string."""
        value = TypedPartitionedRecord(
            partition_type=PartitionType.HOURLY, name="market_data"
        )

        match value.partition_type:
            case PartitionType.HOURLY:
                description = "hourly partitions"
            case PartitionType.DAILY:
                description = "daily partitions"

        self.assertEqual(description, "hourly partitions")
