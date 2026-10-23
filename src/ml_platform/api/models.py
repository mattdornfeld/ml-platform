"""SQLAlchemy mappings for the control-plane PostgreSQL schema."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as POSTGRESQL_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all control-plane ORM mappings."""


class Dataset(Base):
    """One immutable dataset definition."""

    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_type: Mapped[str] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deprecated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class Feature(Base):
    """One immutable feature definition."""

    __tablename__ = "features"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    dtype: Mapped[str] = mapped_column(Text)
    frequency: Mapped[str] = mapped_column(Text)
    keys: Mapped[list[str]] = mapped_column(JSONB)
    validations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    transform: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deprecated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class Label(Base):
    """One immutable label definition."""

    __tablename__ = "labels"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    dtype: Mapped[str] = mapped_column(Text)
    frequency: Mapped[str] = mapped_column(Text)
    keys: Mapped[list[str]] = mapped_column(JSONB)
    validations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    transform: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deprecated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class FeatureDatasetDependency(Base):
    """A declared feature dependency on one dataset version."""

    __tablename__ = "feature_dataset_dependencies"
    __table_args__ = (
        ForeignKeyConstraint(
            ("feature_name", "feature_version"),
            ("features.name", "features.version"),
        ),
        ForeignKeyConstraint(
            ("dataset_name", "dataset_version"),
            ("datasets.name", "datasets.version"),
        ),
    )

    feature_name: Mapped[str] = mapped_column(Text, primary_key=True)
    feature_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_name: Mapped[str] = mapped_column(Text, primary_key=True)
    dataset_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    num_lookback_partitions: Mapped[int] = mapped_column(Integer)


class LabelDatasetDependency(Base):
    """A declared label dependency on one dataset version."""

    __tablename__ = "label_dataset_dependencies"
    __table_args__ = (
        ForeignKeyConstraint(
            ("label_name", "label_version"),
            ("labels.name", "labels.version"),
        ),
        ForeignKeyConstraint(
            ("dataset_name", "dataset_version"),
            ("datasets.name", "datasets.version"),
        ),
    )

    label_name: Mapped[str] = mapped_column(Text, primary_key=True)
    label_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_name: Mapped[str] = mapped_column(Text, primary_key=True)
    dataset_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    num_lookback_partitions: Mapped[int] = mapped_column(Integer)
    num_lookahead_partitions: Mapped[int] = mapped_column(Integer)


class Job(Base):
    """A materialization job and its durable lifecycle state."""

    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True), primary_key=True
    )
    type: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DatasetPartitionRequest(Base):
    """A requested dataset partition revision."""

    __tablename__ = "partition_update_requests"
    __table_args__ = (
        ForeignKeyConstraint(
            ("dataset_name", "dataset_version"),
            ("datasets.name", "datasets.version"),
        ),
    )

    partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    job_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True), ForeignKey("jobs.id")
    )
    dataset_name: Mapped[str] = mapped_column(Text)
    dataset_version: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text)
    callback_token: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class FeaturePartitionRequest(Base):
    """A requested feature partition revision."""

    __tablename__ = "feature_partition_update_requests"
    __table_args__ = (
        ForeignKeyConstraint(
            ("feature_name", "feature_version"),
            ("features.name", "features.version"),
        ),
    )

    partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    job_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True), ForeignKey("jobs.id")
    )
    feature_name: Mapped[str] = mapped_column(Text)
    feature_version: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text)
    callback_token: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LabelPartitionRequest(Base):
    """A requested label partition revision."""

    __tablename__ = "label_partition_update_requests"
    __table_args__ = (
        ForeignKeyConstraint(
            ("label_name", "label_version"),
            ("labels.name", "labels.version"),
        ),
    )

    partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    job_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True), ForeignKey("jobs.id")
    )
    label_name: Mapped[str] = mapped_column(Text)
    label_version: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text)
    callback_token: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DatasetPartition(Base):
    """An Iceberg receipt for one dataset partition revision."""

    __tablename__ = "dataset_partitions"

    partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("partition_update_requests.partition_id"),
        primary_key=True,
    )
    content_hash: Mapped[str] = mapped_column(Text)
    snapshot_id: Mapped[int] = mapped_column(BigInteger)
    row_count: Mapped[int] = mapped_column(BigInteger)
    iceberg_schema: Mapped[dict[str, str]] = mapped_column("schema", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class FeaturePartition(Base):
    """An Iceberg receipt for one feature partition revision."""

    __tablename__ = "feature_partitions"

    partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("feature_partition_update_requests.partition_id"),
        primary_key=True,
    )
    content_hash: Mapped[str] = mapped_column(Text)
    snapshot_id: Mapped[int] = mapped_column(BigInteger)
    row_count: Mapped[int] = mapped_column(BigInteger)
    iceberg_schema: Mapped[dict[str, str]] = mapped_column("schema", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LabelPartition(Base):
    """An Iceberg receipt for one label partition revision."""

    __tablename__ = "label_partitions"

    partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("label_partition_update_requests.partition_id"),
        primary_key=True,
    )
    content_hash: Mapped[str] = mapped_column(Text)
    snapshot_id: Mapped[int] = mapped_column(BigInteger)
    row_count: Mapped[int] = mapped_column(BigInteger)
    iceberg_schema: Mapped[dict[str, str]] = mapped_column("schema", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class FeaturePartitionPlan(Base):
    """Pinned dataset revisions for one feature partition request."""

    __tablename__ = "feature_partition_update_request_dataset_dependencies"

    feature_partition_request_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("feature_partition_update_requests.partition_id"),
        primary_key=True,
    )
    dataset_partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("dataset_partitions.partition_id"),
        primary_key=True,
    )


class LabelPartitionPlan(Base):
    """Pinned dataset revisions for one label partition request."""

    __tablename__ = "label_partition_update_request_dataset_dependencies"

    label_partition_request_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("label_partition_update_requests.partition_id"),
        primary_key=True,
    )
    dataset_partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("dataset_partitions.partition_id"),
        primary_key=True,
    )


class FeaturePartitionDatasetDependency(Base):
    """Completed feature-to-dataset partition lineage edge."""

    __tablename__ = "feature_partition_dataset_dependencies"

    feature_partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("feature_partitions.partition_id"),
        primary_key=True,
    )
    dataset_partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("dataset_partitions.partition_id"),
        primary_key=True,
    )


class LabelPartitionDatasetDependency(Base):
    """Completed label-to-dataset partition lineage edge."""

    __tablename__ = "label_partition_dataset_dependencies"

    label_partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("label_partitions.partition_id"),
        primary_key=True,
    )
    dataset_partition_id: Mapped[UUID] = mapped_column(
        POSTGRESQL_UUID(as_uuid=True),
        ForeignKey("dataset_partitions.partition_id"),
        primary_key=True,
    )
