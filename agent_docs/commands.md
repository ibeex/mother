# Development Commands

## Just Commands

The `justfile` provides shortcuts to `uv` commands. GitHub Actions call `uv` directly.

| Command | Description |
|---------|-------------|
| `just` | Default: install, lint, test |
| `just install` | `uv sync --all-extras` |
| `just lint` | Run codespell, ruff check/format, basedpyright |
| `just format` | Sort imports and format with ruff |
| `just format_md` | Format markdown with rumdl |
| `just test` | `uv run pytest` |
| `just build` | `uv build` |
| `just clean` | Remove build artifacts |
| `just upgrade` | `uv sync --upgrade --all-extras --dev` |
| `just release bump` | Bump version and create git tag |

## Direct uv Commands

```bash
# Run tests
uv run pytest                              # All tests
uv run pytest -s path/to/test.py           # Single test with output

# Dependency management
uv add package_name                        # Add dependency
uv add --dev package_name                  # Add dev dependency
uv sync --upgrade                          # Update all to compatible versions
uv lock --upgrade-package package_name     # Update specific package
uv add package_name@latest                 # Update to latest

# Environment
uv venv && source .venv/bin/activate       # Enter venv shell
uv tool install --editable .               # Install as local tool
```

## Lint Pipeline

`just lint` runs `devtools/lint.py` which executes:

1. `codespell --write-changes` - Fix spelling errors
2. `ruff check --fix` - Lint and auto-fix
3. `ruff format` - Format code
4. `basedpyright --stats` - Type checking
