from __future__ import annotations

import numpy as np
import pandas as pd


def _common_prefix_length(a: str, b: str) -> int:
    """Return the length of the common prefix between two strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def boundary_score(
    prev_name: str,
    curr_name: str,
    prev_phonetic: str,
    curr_phonetic: str,
) -> float:
    """Score how good a batch boundary would be between two adjacent rows.

    Returns a float in [0.0, 1.0]:
      - 0.0 = identical/very similar rows (bad place to split)
      - 1.0 = completely different rows (ideal break point)

    Components:
      - 40% — phonetic code divergence (common prefix ratio)
      - 40% — normalized name prefix divergence
      - 20% — first-character change bonus
    """
    score = 0.0

    # Phonetic divergence
    max_phon = max(len(prev_phonetic), len(curr_phonetic), 1)
    phon_prefix = _common_prefix_length(prev_phonetic, curr_phonetic)
    score += 0.4 * (1.0 - phon_prefix / max_phon)

    # Name prefix divergence
    max_name = max(len(prev_name), len(curr_name), 1)
    name_prefix = _common_prefix_length(prev_name, curr_name)
    score += 0.4 * (1.0 - name_prefix / max_name)

    # First-character change
    if prev_name and curr_name and prev_name[0] != curr_name[0]:
        score += 0.2

    return score


def assign_batches(
    sorted_df: pd.DataFrame,
    estimated_batch_size: int,
    normalized_col: str = "_normalized",
    phonetic_col: str = "_phonetic",
    window_ratio: float = 0.25,
) -> pd.DataFrame:
    """Assign batch_id to a pre-sorted pandas DataFrame.

    Walks through the sorted rows and finds natural break points near
    each ``estimated_batch_size`` boundary using :func:`boundary_score`.
    Similar names are kept together even if the resulting batch is larger
    or smaller than the target.

    Parameters
    ----------
    sorted_df : pd.DataFrame
        Must already be sorted by (phonetic, normalized name).
    estimated_batch_size : int
        Target rows per batch (>= 1).
    normalized_col : str
        Column with normalized names.
    phonetic_col : str
        Column with phonetic codes.
    window_ratio : float
        Fraction of batch size to search on each side of the ideal boundary.

    Returns
    -------
    pd.DataFrame
        Input frame with ``batch_id`` column added.
    """
    n = len(sorted_df)
    if n == 0:
        sorted_df["batch_id"] = pd.Series(dtype=int)
        return sorted_df

    names = sorted_df[normalized_col].values
    phonetics = sorted_df[phonetic_col].values
    batch_ids = np.empty(n, dtype=int)

    batch_id = 1
    batch_start = 0

    while batch_start < n:
        ideal_end = batch_start + estimated_batch_size

        # Last batch: assign all remaining rows
        if ideal_end >= n:
            batch_ids[batch_start:n] = batch_id
            break

        # Search window around the ideal boundary
        window = max(int(estimated_batch_size * window_ratio), 2)
        search_lo = max(batch_start + 1, ideal_end - window)
        search_hi = min(n, ideal_end + window)

        best_break = ideal_end
        best_score = -1.0

        for i in range(search_lo, search_hi + 1):
            if i >= n:
                break
            score = boundary_score(
                names[i - 1], names[i],
                phonetics[i - 1], phonetics[i],
            )
            # Penalise distance from ideal boundary
            distance_penalty = abs(i - ideal_end) / (window + 1)
            adjusted = score - 0.3 * distance_penalty

            if adjusted > best_score:
                best_score = adjusted
                best_break = i

        batch_ids[batch_start:best_break] = batch_id
        batch_id += 1
        batch_start = best_break

    sorted_df = sorted_df.copy()
    sorted_df["batch_id"] = batch_ids
    return sorted_df
