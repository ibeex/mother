## Publishing and releases

### GitHub release readiness

This repository is ready to be published on GitHub.

Suggested first-time steps:

1. Push the repository to GitHub.
2. Confirm the `CI` workflow passes on the default branch.
3. Add a repository description, topics, and screenshots if you want a nicer project page.
4. Create GitHub releases with tags like `v0.8.1` when you want to mark versions.

Example:

```bash
git tag -a v0.8.1 -m "Release v0.8.1"
git push origin main --tags
```

### PyPI status

PyPI publishing is **not configured yet**.

Right now `pyproject.toml` intentionally includes this classifier:

```toml
"Private :: Do Not Upload"
```

That helps avoid accidental uploads.

### If you want to publish to PyPI later

You will need to:

1. Remove the `"Private :: Do Not Upload"` classifier from `pyproject.toml`.
2. Make sure the project name on PyPI is available.
3. Add a dedicated GitHub Actions publish workflow.
4. Configure PyPI trusted publishing for `ibeex/mother`.
5. Build and test distributions before the first upload:

```bash
uv build
```

Optional local verification:

```bash
python -m zipfile -l dist/*.whl
```
