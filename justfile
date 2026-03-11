# justfile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this justfile.

# list all tasks
default:
  @just --list

# Install dependencies
install:
	uv sync --all-extras

# Run linting
lint:
	uv run python devtools/lint.py

# Format code (sort imports and format)
format:
	uv run ruff check --fix --select I src tests devtools
	uv run ruff format src tests devtools

# Format markdown files
format_md files="*.md":
	uv run rumdl fmt --disable MD013 {{ files }}

# Run tests
test:
	uv run pytest

# Upgrade dependencies
upgrade:
	uv sync --upgrade --all-extras --dev

# Build package
build:
	uv build

# Clean build artifacts
clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-find . -type d -name "__pycache__" -exec rm -rf {} +

# Release with version bump
release bump="minor":
        #!/usr/bin/env bash
        set -euo pipefail

        # Check working tree is clean
        if ! git diff-index --quiet HEAD --; then
                echo "Working tree is not clean. Please commit changes first."
                exit 1
        fi

        # Bump version
        uv version --bump {{bump}}

        # Get new version
        NEW_VERSION=$(uv version | awk '{print $2}')

        # Add and commit changes
        git add .
        git commit -m "Released ${NEW_VERSION}"

        # Create tag
        git tag -a "v${NEW_VERSION}" -m "Released version ${NEW_VERSION}"

        echo "Released version ${NEW_VERSION}"
        echo "To push: git push && git push --tags"
