import pandas as pd
import pytest


@pytest.fixture
def sample_names() -> list[str]:
    """Unsorted list of names from the requirements example."""
    return [
        "zzbbbb11",
        "zzzbbbb21",
        "bbbbbb3",
        "bbbbbb4",
        "bbbbbb5",
        "cbbbbb2",
        "cbbbbb11",
        "cbbbbb21",
        "xzbbbbb5",
        "xzbbbb2",
        "aaaa1",
        "aaaa2",
        "aaaa3",
        "bbbbbb1",
        "bbbbbb2",
        "bbbbbb3",
    ]


@pytest.fixture
def sample_df(sample_names: list[str]) -> pd.DataFrame:
    """Pandas DataFrame with a 'name' column from sample_names."""
    return pd.DataFrame({"name": sample_names})
