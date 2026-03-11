# Python Project

Modern Python 3.12+ project using uv for package management.

## Tech Stack

Python 3.12+ · uv · ruff · basedpyright · pytest

## Project Structure

- `src/package_module/` - Main package code
- `tests/` - Test files
- `devtools/` - Development utilities (lint.py)

## Development

```bash
just              # Run install, lint, test
just install      # Install dependencies (uv sync --all-extras)
just lint         # Run codespell, ruff, basedpyright
just format       # Sort imports and format code
just test         # Run pytest
```

Single test: `uv run pytest -s path/to/test.py`

## Standards

- Use `uv` exclusively (never pip/python directly)
- Use `uv add`/`uv remove` for dependencies (don't edit pyproject.toml manually)
- Zero linter/test failures before task completion
- Resolve basedpyright errors during development
- Use `# pyright: ignore` sparingly for unfixable issues

## Code Style

- Full type annotations, use `@override` decorators
- Absolute imports: `from pkg.module import ...`
- Use `pathlib.Path` over file operations
- Use dataclasses for data containers
- Prefer simple functions over classes when sufficient

## Testing

- Test files in `tests/test_*.py`
- Simple inline tests with `## Tests` comment (no pytest imports needed)
- Use `raise AssertionError("msg")` not `assert False`

## Detailed Documentation

See `agent_docs/` for details:

- `commands.md` - Full command reference and uv workflows
- `testing.md` - Testing conventions and patterns
- `python.md` - Python coding guidelines and style
