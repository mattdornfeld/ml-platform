# Python code quality

This repository follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Run the complete local gate with:

```shell
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
uv run --locked pytest -q
```

`pre-commit install` enables the same Ruff and Mypy checks before commits.

## Enforced by Ruff

Ruff enforces Google-aligned import organization, 80-character formatting, naming,
Google-style docstrings, annotation presence, exception handling, complexity,
security, bug detection, and modernization. The enabled rule families are declared
explicitly in `pyproject.toml`.

The formatter owns whitespace, indentation, quotes, commas, and line wrapping.
Ruff rules that conflict with formatter output are excluded. `E501` is excluded
because the formatter deliberately preserves unavoidable long URLs, SQL, and Spark
identifiers, which Google allows as line-length exceptions. Test modules are exempt
from docstring, assertion-security, and private-contract rules because the Google
guide does not require test-module docstrings and these patterns are intrinsic to
pytest and contract testing.

## Gaps

Ruff cannot enforce every Google recommendation, including API design, appropriate
abstraction boundaries, naming semantics, meaningful docstring content, and review
judgment. Mypy runs in strict mode for `src` to enforce type safety beyond Ruff's
annotation checks. Human review remains responsible for the remaining guidance.
