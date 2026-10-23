"""Public errors raised while validating shared resource definitions."""


class DomainConfigurationError(ValueError):
    """A resource definition violates a shared domain invariant."""


class TransformDependencyConfigurationError(DomainConfigurationError):
    """A transform definition declares ambiguous input datasets or keys."""


class MarketSymbolConfigurationError(DomainConfigurationError):
    """A market-data definition has no usable ticker symbols."""
