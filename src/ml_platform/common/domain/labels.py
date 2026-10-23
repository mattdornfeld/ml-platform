"""Label definition models shared by API clients and Spark workers."""

from pydantic import Field

from ml_platform.common.domain.dependencies import LabelDatasetDependency
from ml_platform.common.domain.resources import TransformConfig
from ml_platform.common.types import LabelId, LabelName


class LabelConfig(TransformConfig[LabelName]):
    """A complete immutable label resource definition."""

    name: LabelName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase label name.",
    )
    datasets: list[LabelDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into label transforms by dataset name.",
    )

    @property
    def id(self) -> LabelId:
        """Return the canonical immutable label identifier."""
        return LabelId(f"{self.name}_v{self.version}")


class TripleBarrierConfig(LabelConfig):
    """Triple-barrier label strategy and its volatility-scaled parameters."""

    horizon_bars: int = Field(
        ge=1,
        description="Maximum number of bars before the vertical barrier expires.",
    )
    volatility_lookback_bars: int = Field(
        ge=2, description="Trailing bars used to estimate volatility."
    )
    profit_taking_multiplier: float = Field(
        gt=0.0,
        description="Volatility multiple that defines the profit-taking barrier.",
    )
    stop_loss_multiplier: float = Field(
        gt=0.0,
        description="Volatility multiple that defines the stop-loss barrier.",
    )
