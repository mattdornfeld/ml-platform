"""Typed Hydra configuration loading for command-line data workflows."""

import argparse
from datetime import date
from functools import wraps
from pathlib import Path
from enum import StrEnum
from typing import Callable, Literal, TypeVar

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_DIR = Path("configs")
VARIANT_CONFIG_DIR = CONFIG_DIR / "variants"
FEATURE_CONFIG_DIR = CONFIG_DIR / "features"
FEATURE_APPLICATION_CONFIG_DIR = CONFIG_DIR / "feature_materialization"
DEFAULT_VARIANT_NAME = "sp500"
Result = TypeVar("Result")


class FeatureKey(StrEnum):
    """Supported logical keys for feature output rows."""

    SYMBOL = "symbol"


class FeatureValidations(BaseModel):
    """Configurable validation requirements for returned feature DataFrames."""

    model_config = ConfigDict(extra="forbid")

    min_rows_per_partition: int = Field(default=1, ge=1)
    require_finite_feature_value: bool = True
    require_feature_keys_from_input: bool = True


class MarketDataConfig(BaseModel):
    """Validated configuration for downloading and storing hourly market data."""

    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(default_factory=lambda: ["AAPL"], min_length=1)
    start_date: date
    end_date: date

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, symbols: list[str]) -> list[str]:
        """Normalize and validate the configured Yahoo Finance symbols."""
        normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not normalized_symbols:
            raise ValueError("Provide at least one non-empty market symbol.")
        return normalized_symbols

    @model_validator(mode="after")
    def validate_date_range(self) -> "MarketDataConfig":
        """Require an end date after the required inclusive start date."""
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date.")
        return self


class LongShortLabelConfig(BaseModel):
    """Validated parameters for Spark-generated triple-barrier target labels."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(ge=1)
    horizon_bars: int = Field(ge=1)
    volatility_lookback_bars: int = Field(ge=2)
    profit_taking_multiplier: float = Field(gt=0.0)
    stop_loss_multiplier: float = Field(gt=0.0)


class FeatureConfig(BaseModel):
    """Feature metadata and its mapped Spark transformation entrypoint."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(ge=1)
    dtype: Literal["double", "float", "int", "bigint"]
    frequency: Literal["hourly", "daily"]
    feature_key: FeatureKey = FeatureKey.SYMBOL
    datasets: list["DatasetMetadata"] = Field(min_length=1)
    transform: str = Field(pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    parameters: dict[str, int | float | str | bool] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    validations: FeatureValidations = Field(default_factory=FeatureValidations)

    @field_validator("datasets")
    @classmethod
    def require_unique_datasets(
        cls, datasets: list["DatasetMetadata"]
    ) -> list["DatasetMetadata"]:
        """Require at most one declaration for each upstream dataset."""
        if len({dataset.dataset_id for dataset in datasets}) != len(datasets):
            raise ValueError("Feature datasets must have unique dataset_id values.")
        return datasets


class DatasetMetadata(BaseModel):
    """Declarative identity and preceding-partition extent for one dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    num_lookback_partitions: int = Field(ge=0)


class FeatureMaterializationConfig(BaseModel):
    """Application settings for concurrent feature backfills."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_spark_jobs: int = Field(ge=1)


class VariantsConfig(BaseModel):
    """Hydra-composed configuration for one named data-workflow variant."""

    model_config = ConfigDict(extra="forbid")

    market_data: MarketDataConfig
    labels: LongShortLabelConfig


def load_variants_config(variant_name: str) -> VariantsConfig:
    """Compose and validate one named Hydra variant from ``configs/variants``."""
    _named_config_path(VARIANT_CONFIG_DIR, variant_name)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        composed_config = compose(config_name=f"variants/{variant_name}")
    resolved_config = OmegaConf.to_container(composed_config, resolve=True)
    return VariantsConfig.model_validate(resolved_config)


def load_feature_config(feature_name: str) -> FeatureConfig:
    """Load and validate one feature configuration from ``configs/features``."""
    _named_config_path(FEATURE_CONFIG_DIR, feature_name)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        composed_config = compose(config_name=f"features/{feature_name}")
    resolved_config = OmegaConf.to_container(composed_config, resolve=True)
    return FeatureConfig.model_validate(resolved_config)


def list_feature_config_names() -> list[str]:
    """Return every checked-in feature configuration name in deterministic order."""
    return sorted(config_path.stem for config_path in FEATURE_CONFIG_DIR.glob("*.yaml"))


def load_feature_materialization_config(config_name: str) -> FeatureMaterializationConfig:
    """Load the feature-materialization application configuration."""
    _named_config_path(FEATURE_APPLICATION_CONFIG_DIR, config_name)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        composed_config = compose(config_name=f"feature_materialization/{config_name}")
    resolved_config = OmegaConf.to_container(composed_config, resolve=True)
    return FeatureMaterializationConfig.model_validate(resolved_config)


def parse_variant_args() -> argparse.Namespace:
    """Parse the Hydra variant name used by a data-workflow entrypoint."""
    parser = argparse.ArgumentParser(description="Run a named data-workflow variant.")
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT_NAME,
        help="Name of a YAML configuration in configs/variants.",
    )
    return parser.parse_args()


def parse_feature_args() -> argparse.Namespace:
    """Parse the feature configuration name for the feature entrypoint."""
    parser = argparse.ArgumentParser(description="Materialize configured market-data features.")
    parser.add_argument("--config", default="default", help="Name of YAML configuration in configs/feature_materialization.")
    return parser.parse_args()


def with_variants_config(function: Callable[[VariantsConfig], Result]) -> Callable[[], Result]:
    """Inject the CLI-selected ``VariantsConfig`` into a zero-argument entrypoint."""

    @wraps(function)
    def wrapped() -> Result:
        args = parse_variant_args()
        return function(load_variants_config(args.variant))

    return wrapped


def _named_config_path(config_directory: Path, config_name: str) -> Path:
    """Resolve a named YAML configuration without allowing directory traversal."""
    if not config_name or Path(config_name).name != config_name:
        raise ValueError("Configuration names must not include a path.")
    config_path = config_directory / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    return config_path
