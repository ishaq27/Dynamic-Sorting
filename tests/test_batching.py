import pandas as pd
import pytest

from snowpark_batch.batching import assign_batches, boundary_score
from snowpark_batch.normalize import normalize_series
from snowpark_batch.phonetic import phonetic_series


class TestBoundaryScore:
    def test_identical_rows_score_zero(self):
        assert boundary_score("abc", "abc", "ABK", "ABK") == 0.0

    def test_completely_different_rows_high_score(self):
        score = boundary_score("aaa", "zzz", "A", "S")
        assert score > 0.7

    def test_first_char_change_bonus(self):
        score_same = boundary_score("abc", "abd", "ABK", "ABT")
        score_diff = boundary_score("abc", "zbd", "ABK", "SBT")
        assert score_diff > score_same

    def test_score_range(self):
        score = boundary_score("hello", "world", "HL", "WRLT")
        assert 0.0 <= score <= 1.0


class TestAssignBatches:
    def _prepare_df(self, names: list[str]) -> pd.DataFrame:
        """Create a sorted DataFrame with _normalized and _phonetic columns."""
        df = pd.DataFrame({"name": names})
        df["_normalized"] = normalize_series(
            df["name"],
            transliterate_unicode=True,
            lowercase=True,
            remove_punctuation=True,
            remove_suffixes=True,
        )
        df["_phonetic"] = phonetic_series(df["_normalized"])
        df = df.sort_values(
            ["_phonetic", "_normalized"]
        ).reset_index(drop=True)
        return df

    def test_requirements_example(self, sample_names):
        """Test the exact example from the requirements.

        With estimated_batch_size=8 and 16 scattered names, the algorithm
        should group similar names together and produce reasonable batches.
        """
        df = self._prepare_df(sample_names)
        result = assign_batches(df, estimated_batch_size=8)

        assert "batch_id" in result.columns
        # All rows assigned
        assert result["batch_id"].notna().all()
        # At least 1 batch
        assert result["batch_id"].min() >= 1

        # Verify similar names are in the same batch
        batch_for_name = dict(zip(result["_normalized"], result["batch_id"]))

        # All "aaaa*" should be in the same batch
        aaaa_batches = {
            batch_for_name[n]
            for n in batch_for_name
            if n.startswith("aaaa")
        }
        assert len(aaaa_batches) == 1, f"aaaa* split across batches: {aaaa_batches}"

        # All "bbbbbb*" should be in the same batch
        bbb_batches = {
            batch_for_name[n]
            for n in batch_for_name
            if n.startswith("bbbbbb")
        }
        assert len(bbb_batches) == 1, f"bbbbbb* split across batches: {bbb_batches}"

    def test_empty_dataframe(self):
        df = pd.DataFrame({"_normalized": pd.Series(dtype=str), "_phonetic": pd.Series(dtype=str)})
        result = assign_batches(df, estimated_batch_size=5)
        assert "batch_id" in result.columns
        assert len(result) == 0

    def test_single_row(self):
        df = self._prepare_df(["hello"])
        result = assign_batches(df, estimated_batch_size=5)
        assert len(result) == 1
        assert result["batch_id"].iloc[0] == 1

    def test_batch_size_one(self):
        df = self._prepare_df(["aaa", "bbb", "ccc"])
        result = assign_batches(df, estimated_batch_size=1)
        assert result["batch_id"].nunique() >= 2

    def test_batch_size_larger_than_data(self):
        df = self._prepare_df(["aaa", "bbb", "ccc"])
        result = assign_batches(df, estimated_batch_size=100)
        assert result["batch_id"].nunique() == 1

    def test_all_identical_names(self):
        df = self._prepare_df(["same"] * 20)
        result = assign_batches(df, estimated_batch_size=8)
        # Should produce batches (boundary scores will be 0, falls back to ideal_end)
        assert result["batch_id"].notna().all()
        assert result["batch_id"].min() >= 1

    def test_no_record_split(self, sample_names):
        """Verify that identical normalized names are never split across batches."""
        df = self._prepare_df(sample_names)
        result = assign_batches(df, estimated_batch_size=8)

        # Group by normalized name and check each group has exactly 1 batch_id
        for name, group in result.groupby("_normalized"):
            batch_ids = group["batch_id"].unique()
            assert len(batch_ids) == 1, (
                f"Name {name!r} split across batches: {batch_ids}"
            )

    def test_batches_are_sequential(self, sample_names):
        df = self._prepare_df(sample_names)
        result = assign_batches(df, estimated_batch_size=8)
        batch_ids = result["batch_id"].unique()
        # Batch IDs should be 1, 2, 3, ... (no gaps)
        expected = list(range(1, len(batch_ids) + 1))
        assert sorted(batch_ids) == expected
