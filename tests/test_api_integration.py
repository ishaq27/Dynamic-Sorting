"""Integration tests for the full batch_dataframe() pipeline.

These tests use pandas DataFrames directly (bypassing Snowpark) to validate
the normalization -> sort -> batch flow end-to-end without needing a
Snowflake connection.  The Snowpark-specific join logic is thin and
straightforward; the core algorithm is tested here via the lower-level
functions.
"""

import pandas as pd
import pytest

from snowpark_batch.batching import assign_batches
from snowpark_batch.normalize import normalize_series
from snowpark_batch.phonetic import phonetic_series


# Default flags used by integration tests (full normalization)
_NORM_FLAGS = dict(
    replace_symbols=True,
    transliterate_unicode=True,
    lowercase=True,
    remove_punctuation=True,
    remove_suffixes=True,
)


def _run_pipeline(
    names: list[str],
    estimated_batch_size: int,
    window_ratio: float = 0.25,
    norm_flags: dict | None = None,
) -> pd.DataFrame:
    """Simulate the batch_dataframe() pipeline using pure pandas."""
    flags = norm_flags if norm_flags is not None else _NORM_FLAGS
    df = pd.DataFrame({"name": names})
    df["name_norm"] = normalize_series(df["name"], **flags)
    df["_phonetic"] = phonetic_series(df["name_norm"])
    df = df.sort_values(
        ["_phonetic", "name_norm"],
    ).reset_index(drop=True)
    df["row_id"] = range(1, len(df) + 1)
    df = assign_batches(
        df,
        estimated_batch_size,
        normalized_col="name_norm",
        window_ratio=window_ratio,
    )
    return df


class TestFullPipeline:
    def test_requirements_example(self, sample_names):
        result = _run_pipeline(sample_names, estimated_batch_size=8)

        assert "batch_id" in result.columns
        assert "row_id" in result.columns
        assert "name_norm" in result.columns
        assert len(result) == len(sample_names)

        # Original name column is preserved
        assert "name" in result.columns

        # row_id should be sequential 1..n
        assert list(result["row_id"]) == list(range(1, len(sample_names) + 1))

        # All batches should have at least 1 row
        for _, group in result.groupby("batch_id"):
            assert len(group) >= 1

    def test_unicode_names_batching(self):
        names = [
            "Müller AG",
            "Mueller Inc",
            "Muller LLC",
            "Schmidt & Co",
            "Schmitt Ltd",
            "Café Restaurant",
            "Cafe Bistro",
            "Straße Bar",
            "Strasse Bar",
        ]
        result = _run_pipeline(names, estimated_batch_size=5)

        assert len(result) == len(names)
        assert result["batch_id"].min() >= 1
        # Verify normalized column exists and original is preserved
        assert "name_norm" in result.columns
        assert "name" in result.columns

    def test_all_same_phonetic(self):
        # Names that all sound similar
        names = ["Smith", "Smyth", "Smithe", "Smythe", "Smithh"]
        result = _run_pipeline(names, estimated_batch_size=3)

        assert len(result) == 5
        assert result["batch_id"].notna().all()

    def test_empty_input(self):
        result = _run_pipeline([], estimated_batch_size=8)
        assert len(result) == 0
        assert "batch_id" in result.columns

    def test_large_batch_size(self):
        names = [f"name{i}" for i in range(5)]
        result = _run_pipeline(names, estimated_batch_size=100)
        assert result["batch_id"].nunique() == 1

    def test_batch_size_one(self):
        names = ["alpha", "beta", "gamma"]
        result = _run_pipeline(names, estimated_batch_size=1)
        # With distinct names and batch_size=1, should have multiple batches
        assert result["batch_id"].nunique() >= 2

    def test_no_normalization_flags(self):
        """When no flags are set, only whitespace cleanup happens."""
        names = ["Müller AG", "Mueller Inc", "Café LLC"]
        result = _run_pipeline(
            names,
            estimated_batch_size=10,
            norm_flags={},
        )
        # Original names should be mostly preserved (only whitespace cleanup)
        norm_values = set(result["name_norm"])
        assert "Müller AG" in norm_values
        assert "Mueller Inc" in norm_values
        assert "Café LLC" in norm_values

    def test_selective_normalization(self):
        """Only lowercase enabled — other steps skipped."""
        names = ["Müller & Co", "HELLO WORLD"]
        result = _run_pipeline(
            names,
            estimated_batch_size=10,
            norm_flags={"lowercase": True},
        )
        norm_values = set(result["name_norm"])
        # Unicode kept, symbols kept, but lowercased
        assert "müller & co" in norm_values
        assert "hello world" in norm_values
