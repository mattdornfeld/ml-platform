# Project conventions

- Use `/Users/matthewdornfeld/.virtualenvs/pytorch-tutorial/bin/python` as this repository's Python interpreter.
- Use absolute imports for all Python modules in this repository.
- Do not preserve obsolete schemas or paths with compatibility shims, symlinks, or fallbacks. Migrate data and update code to the current schema instead.
- When the user gives a goal, attempt straightforward fixes for errors encountered while pursuing it without stopping to ask for permission. Ask only when progress depends on a consequential decision or unavailable information.
- Put values derived from materializer instance configuration behind descriptive private instance properties rather than recomputing them inside workflow methods.
- Call materializer instance behavior through `self`, including transform instantiation, rather than naming the concrete class.
- Break complex nested transform and DataFrame expressions into named steps or multiline keyword arguments.
- Use descriptive variable names that explain the data or role of each intermediate value.
- Prefer frozen dataclasses for materializer instance properties. Avoid mutable materializer instance state; pass or recompute transient workflow data instead of assigning mutable caches to `self`.
- Do not infer or derive runtime behavior, schemas, partitioning, or other implementation details from metadata without explicit user approval. Prefer static declarations and strong types.
- Do not introduce optional properties, arguments, or return types without explicit user approval. Prefer required, statically typed interfaces.
