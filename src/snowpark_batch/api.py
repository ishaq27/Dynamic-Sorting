from __future__ import annotations

import pandas as pd

from snowpark_batch.batching import assign_batches
from snowpark_batch.normalize import normalize_series
from snowpark_batch.phonetic import phonetic_series


def _is_snowpark_dataframe(df: object) -> bool:
    """Check if *df* is a Snowpark DataFrame without importing Snowpark eagerly."""
    cls_name = type(df).__name__
    mod = type(df).__module__ or ""
    return cls_name == "DataFrame" and "snowflake" in mod


def _batch_pandas(
    df: pd.DataFrame,
    key_column: str,
    estimated_batch_size: int,
    columns: list[str] | None,
    sort_ascending: bool,
    similarity_window_ratio: float,
    norm_flags: dict,
) -> pd.DataFrame:
    """Core batching logic on a pandas DataFrame."""
    if key_column not in df.columns:
        raise ValueError(
            f"key_column {key_column!r} not found in DataFrame columns: {list(df.columns)}"
        )
    if estimated_batch_size < 1:
        raise ValueError("estimated_batch_size must be >= 1")
    if not (0 < similarity_window_ratio <= 1):
        raise ValueError("similarity_window_ratio must be in (0, 1]")

    norm_col_name = f"{key_column}_norm"

    # Work on a copy so the original is untouched
    pdf = df.copy()
    pdf["_uid"] = range(len(pdf))

    # ── normalize & phonetic encode ─────────────────────────────────
    pdf[norm_col_name] = normalize_series(pdf[key_column], **norm_flags)
    pdf["_phonetic"] = phonetic_series(pdf[norm_col_name])

    # ── sort ────────────────────────────────────────────────────────
    pdf = pdf.sort_values(
        ["_phonetic", norm_col_name],
        ascending=sort_ascending,
    ).reset_index(drop=True)

    # ── row_id (1-based) ───────────────────────────────────────────
    pdf["row_id"] = range(1, len(pdf) + 1)

    # ── assign batches ─────────────────────────────────────────────
    pdf = assign_batches(
        pdf,
        estimated_batch_size,
        normalized_col=norm_col_name,
        window_ratio=similarity_window_ratio,
    )

    # ── drop internal helper columns ───────────────────────────────
    pdf = pdf.drop(columns=["_uid", "_phonetic"])

    # ── select requested columns ───────────────────────────────────
    if columns is not None:
        extra = {"batch_id", "row_id", norm_col_name}
        keep = [c for c in columns if c not in extra]
        keep += [norm_col_name, "batch_id", "row_id"]
        pdf = pdf[keep]

    return pdf


def _batch_snowpark(
    df,
    key_column: str,
    estimated_batch_size: int,
    columns: list[str] | None,
    sort_ascending: bool,
    similarity_window_ratio: float,
    norm_flags: dict,
):
    """Snowpark-specific path: pull key column to client, batch, join back."""
    from snowflake.snowpark import functions as F
    from snowflake.snowpark import Window

    col_names = [c.name for c in df.schema.fields]
    if key_column.upper() not in (c.upper() for c in col_names):
        raise ValueError(
            f"key_column {key_column!r} not found in DataFrame columns: {col_names}"
        )
    if estimated_batch_size < 1:
        raise ValueError("estimated_batch_size must be >= 1")
    if not (0 < similarity_window_ratio <= 1):
        raise ValueError("similarity_window_ratio must be in (0, 1]")

    norm_col_name = f"{key_column}_norm".upper()
    session = df._session  # noqa: SLF001

    # ── add a unique row identifier ─────────────────────────────────
    uid_col = "_snowpark_batch_uid"
    df_with_uid = df.with_column(
        uid_col,
        F.row_number().over(Window.order_by(F.lit(1))),
    )

    # ── pull only the key column + uid to the client ────────────────
    pdf: pd.DataFrame = df_with_uid.select(uid_col, key_column).to_pandas()
    uid_upper = uid_col.upper()
    key_upper = key_column.upper()

    # ── normalize & phonetic encode ─────────────────────────────────
    pdf["_normalized"] = normalize_series(pdf[key_upper], **norm_flags)
    pdf["_phonetic"] = phonetic_series(pdf["_normalized"])

    # ── sort ────────────────────────────────────────────────────────
    pdf = pdf.sort_values(
        ["_phonetic", "_normalized"],
        ascending=sort_ascending,
    ).reset_index(drop=True)

    # ── row_id (1-based) ───────────────────────────────────────────
    pdf["row_id"] = range(1, len(pdf) + 1)

    # ── assign batches ─────────────────────────────────────────────
    pdf = assign_batches(
        pdf,
        estimated_batch_size,
        window_ratio=similarity_window_ratio,
    )

    # ── join back to Snowpark ──────────────────────────────────────
    result_pdf = pdf[[uid_upper, "_normalized", "batch_id", "row_id"]]
    result_pdf = result_pdf.rename(columns={"_normalized": norm_col_name})
    batch_sdf = session.create_dataframe(result_pdf)

    joined = df_with_uid.join(
        batch_sdf,
        df_with_uid[uid_col] == batch_sdf[uid_upper],
    )
    joined = joined.drop(uid_col).drop(uid_upper)

    # ── select requested columns ───────────────────────────────────
    if columns is not None:
        extra = {"BATCH_ID", "ROW_ID", norm_col_name}
        keep = [c for c in columns if c.upper() not in extra]
        keep += [norm_col_name, "BATCH_ID", "ROW_ID"]
        joined = joined.select(keep)

    return joined


def batch_dataframe(
    df,
    key_column: str,
    estimated_batch_size: int,
    columns: list[str] | None = None,
    sort_ascending: bool = True,
    similarity_window_ratio: float = 0.25,
    *,
    replace_symbols: bool = False,
    transliterate_unicode: bool = False,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_suffixes: bool = False,
):
    """Add ``batch_id``, ``row_id``, and ``{key_column}_norm`` columns.

    Works with both **pandas DataFrames** and **Snowpark DataFrames**.

    Applies only the normalization steps you enable via boolean flags.
    The normalized value is stored in a new column named
    ``{key_column}_norm`` — the original column is never modified.
    Sorting and batching use the normalized column.

    Parameters
    ----------
    df : pd.DataFrame | SnowparkDataFrame
        Input DataFrame (pandas or Snowpark).
    key_column : str
        Column containing names to normalize and group.
    estimated_batch_size : int
        Target rows per batch (>= 1).  Actual batches may be larger or
        smaller to keep similar names together.
    columns : list[str] | None
        Columns to keep in the output.  ``None`` keeps all original columns.
    sort_ascending : bool
        Sort direction (default ``True`` = A-Z).
    similarity_window_ratio : float
        Fraction of *estimated_batch_size* to search on each side of the
        ideal boundary for natural breaks.  Default ``0.25``.
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

    Returns
    -------
    pd.DataFrame | SnowparkDataFrame
        Original columns + ``batch_id`` (int) + ``row_id`` (int)
        + ``{key_column}_norm`` (str).
    """
    norm_flags = dict(
        replace_symbols=replace_symbols,
        transliterate_unicode=transliterate_unicode,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_suffixes=remove_suffixes,
    )

    if _is_snowpark_dataframe(df):
        return _batch_snowpark(
            df, key_column, estimated_batch_size,
            columns, sort_ascending, similarity_window_ratio, norm_flags,
        )

    if isinstance(df, pd.DataFrame):
        return _batch_pandas(
            df, key_column, estimated_batch_size,
            columns, sort_ascending, similarity_window_ratio, norm_flags,
        )

    raise TypeError(
        f"df must be a pandas DataFrame or Snowpark DataFrame, got {type(df).__name__}"
    )
