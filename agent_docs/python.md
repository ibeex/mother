# Python Coding Guidelines

Target: Python 3.12-3.14 with modern practices.

## Type Annotations

- Full type annotations on all functions
- Modern syntax: `str | None`, `dict[str, int]`, `list[str]` (never `Optional`)
- Use `@override` decorators when overriding methods
- Use generics where appropriate

## Imports

- Absolute imports: `from pkg.module import ...` (not relative)
- Import from correct modules:
  - `collections.abc` for abstract types (Mapping, Sequence, etc.)
  - `typing_extensions` for backported features

## String Operations

Don't use `re` module for trivial operations. Prefer:

- `str.split()`, `str.join()`
- `str.replace()`
- `in`, `str.startswith()`, `str.endswith()`

## File Operations

- Use `pathlib.Path` over `os.path`
- Use `Path.read_text()` / `Path.write_text()` over `open()`

## Data Structures

- Use `dataclasses` for data containers
- Use `StrEnum` for string enums with lowercase values (for JSON protocols)

## Multi-line Strings

```python
from textwrap import dedent

text = dedent("""
    First line
    Second line
""").strip('\n')
```

## Comments

- Use only when code intent isn't obvious
- Explain WHY, not WHAT
- Keep concise and production-ready
- Avoid: obvious descriptions, decorative headings, numbered steps, emojis

## Docstrings

- Triple quotes on own lines
- Explain WHY not WHAT
- Use `backticks` for variables, ``` for code blocks
- Public methods need docstrings; internal ones only if unclear
- Document rationale and pitfalls, not obvious behavior

## Best Practices

- Avoid trivial wrapper functions
- Don't make a class when a simple function suffices
- Use `# pyright: ignore[reportUnusedParameter]` for intentionally unused params
- Preserve existing comments, pydocs, and log statements when editing
- Ask before globally disabling lint rules
