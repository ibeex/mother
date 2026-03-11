# Testing Conventions

## Test Locations

- **Complex tests**: `tests/test_*.py`
- **Simple inline tests**: Within source files using `## Tests` comment

## Running Tests

```bash
just test                          # Run all tests
uv run pytest                      # Same as above
uv run pytest -s path/to/test.py   # Single test with output
```

## Test Configuration

From `pyproject.toml`:

- `python_files = ["*.py"]` - All .py files can contain tests
- `testpaths = ["src", "tests"]` - Search both src and tests

## Inline Tests

For simple module tests, add at end of file:

```python
## Tests

def test_my_function():
    result = my_function(5)
    assert result == 10
```

No pytest imports needed for simple tests.

## Assertions

- Use `raise AssertionError("descriptive message")` instead of `assert False`
- Keep assertions simple: `assert x == 5` (no redundant messages)
- Assert one thing per test when possible

## What NOT to Do

- Don't create throwaway test files
- Don't use `if __name__ == "__main__"` for running tests
- Don't write trivial tests that just verify Python works
- Don't add pytest imports for simple inline tests
