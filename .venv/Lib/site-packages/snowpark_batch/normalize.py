from __future__ import annotations

import re

import pandas as pd
from unidecode import unidecode

from snowpark_batch.constants import ENTITY_SUFFIX_PATTERN, SYMBOL_REPLACEMENTS

_MULTI_SPACE = re.compile(r"\s{2,}")
_PUNCTUATION = re.compile(r"[.,\-'\"!?;:()\[\]{}/\\#*]")


def normalize_name(
    raw: str | None,
    *,
    replace_symbols: bool = False,
    transliterate_unicode: bool = False,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_suffixes: bool = False,
) -> str:
    """Normalize a company/person name with configurable steps.

    All normalization steps default to ``False``.  Only the steps you
    explicitly enable will be applied.  Basic cleanup (strip whitespace,
    collapse multiple spaces) always runs.

    Parameters
    ----------
    raw : str | None
        The raw name to normalize.
    replace_symbols : bool
        Replace symbol characters (&, +, @) with word equivalents.
    transliterate_unicode : bool
        Transliterate Unicode to ASCII (e.g. Müller → Muller).
    lowercase : bool
        Convert to lowercase.
    remove_punctuation : bool
        Remove punctuation characters.
    remove_suffixes : bool
        Remove entity suffixes (LLC, Inc, Ltd, ...).
    """
    if raw is None:
        return ""

    text = str(raw).strip()
    if not text:
        return ""

    # Symbol replacement
    if replace_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)

    # Unicode -> ASCII
    if transliterate_unicode:
        text = unidecode(text)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation
    if remove_punctuation:
        text = _PUNCTUATION.sub(" ", text)

    # Collapse multiple spaces
    text = _MULTI_SPACE.sub(" ", text).strip()

    # Remove entity suffixes
    if remove_suffixes:
        text = ENTITY_SUFFIX_PATTERN.sub("", text).strip()
        # Final collapse (suffix removal may leave trailing spaces)
        text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def normalize_series(
    series: pd.Series,
    *,
    replace_symbols: bool = False,
    transliterate_unicode: bool = False,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_suffixes: bool = False,
) -> pd.Series:
    """Apply normalize_name to every element in a pandas Series.

    Passes all normalization flags through to :func:`normalize_name`.
    """
    return series.apply(
        normalize_name,
        replace_symbols=replace_symbols,
        transliterate_unicode=transliterate_unicode,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_suffixes=remove_suffixes,
    )
