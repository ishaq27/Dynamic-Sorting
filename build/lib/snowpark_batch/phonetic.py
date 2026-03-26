from __future__ import annotations

import jellyfish
import pandas as pd


def phonetic_code(name: str) -> str:
    """Return the Metaphone phonetic code for a normalized name.

    Metaphone produces variable-length codes that preserve more phonetic
    structure than Soundex, enabling meaningful prefix comparison for
    batch boundary detection.
    """
    if not name:
        return ""
    return jellyfish.metaphone(name)


def phonetic_series(series: pd.Series) -> pd.Series:
    """Apply phonetic_code to every element in a pandas Series."""
    return series.apply(phonetic_code)
