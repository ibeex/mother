# Development

## Setting Up uv

This project is set up to use [uv](https://docs.astral.sh/uv/) to manage Python and
dependencies. First, be sure you
[have uv installed](https://docs.astral.sh/uv/getting-started/installation/).

Then fork this repository (having your own fork will make it easier to contribute)
and [clone it](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

## Basic Developer Workflows

The `justfile` simply offers shortcuts to `uv` commands for developer convenience.
(For clarity, the CI workflow doesn't use the justfile and just calls `uv` directly.)

```shell
# First, install all dependencies and set up your virtual environment.
# This simply runs `uv sync --all-extras` to install all packages,
# including dev dependencies and optional dependencies.
just install

# Run uv sync, lint, and test:
just

# Build wheel:
just build

# Linting:
just lint

# Run tests:
just test

# Delete all the build artifacts:
just clean

# Upgrade dependencies to compatible versions:
just upgrade

# To run tests by hand:
uv run pytest   # all tests
uv run pytest -s src/module/some_file.py  # one test, showing outputs

# Build and install current dev executables, to let you use your dev copies
# as local tools:
uv tool install --editable .

# Dependency management directly with uv:
# Add a new dependency:
uv add package_name
# Add a development dependency:
uv add --dev package_name
# Update to latest compatible versions (including dependencies on git repos):
uv sync --upgrade
# Update a specific package:
uv lock --upgrade-package package_name
# Update dependencies on a package:
uv add package_name@latest

# Run a shell within the Python environment:
uv venv
source .venv/bin/activate
```

See [uv docs](https://docs.astral.sh/uv/) for details.

## Documentation

- [uv docs](https://docs.astral.sh/uv/)

- [basedpyright docs](https://docs.basedpyright.com/latest/)

* * *

*This file was built with
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
