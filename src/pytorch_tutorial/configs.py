"""Typed Hydra configuration loading for command-line data workflows."""

import argparse
from datetime import date
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_DIR = Path("configs")
VARIANT_CONFIG_DIR = CONFIG_DIR / "variants"
DEFAULT_VARIANT_NAME = "sp500"
Result = TypeVar("Result")


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

    horizon_bars: int = Field(ge=1)
    volatility_lookback_bars: int = Field(ge=2)
    profit_taking_multiplier: float = Field(gt=0.0)
    stop_loss_multiplier: float = Field(gt=0.0)


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


def parse_variant_args() -> argparse.Namespace:
    """Parse the Hydra variant name used by a data-workflow entrypoint."""
    parser = argparse.ArgumentParser(description="Run a named data-workflow variant.")
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT_NAME,
        help="Name of a YAML configuration in configs/variants.",
    )
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
