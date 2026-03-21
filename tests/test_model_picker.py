"""Tests for shared picker search in the model picker."""

from mother.model_picker import filter_available_models


def test_filter_available_models_supports_fuzzy_subsequence_matches() -> None:
    available_models = [
        ("local_1", "local_1 — local 1"),
        ("local_2", "local_2 — local 2"),
        ("local_3", "local_3 — local 3"),
    ]

    matches = filter_available_models("lo3", available_models)

    assert [model_id for model_id, _ in matches] == ["local_3"]
