#!/usr/bin/env python3
"""Example: run the fuzzy batching pipeline on sample data (no Snowflake needed).

This script demonstrates the normalization, phonetic encoding, and
batch assignment using pure pandas — the same logic that runs inside
``batch_dataframe()`` when connected to Snowflake.
"""

import pandas as pd

from snowpark_batch.batching import assign_batches
from snowpark_batch.normalize import normalize_series
from snowpark_batch.phonetic import phonetic_series

# ── Sample data (scattered, needs normalization) ────────────────────
raw_names = [
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

ESTIMATED_BATCH_SIZE = 8

# ── Build DataFrame ─────────────────────────────────────────────────
df = pd.DataFrame({"name": raw_names})

# ── Normalize ───────────────────────────────────────────────────────
df["_normalized"] = normalize_series(df["name"])
df["_phonetic"] = phonetic_series(df["_normalized"])

# ── Sort by phonetic + normalized name ──────────────────────────────
df = df.sort_values(["_phonetic", "_normalized"]).reset_index(drop=True)

# ── Assign row_id ───────────────────────────────────────────────────
df["row_id"] = range(1, len(df) + 1)

# ── Assign batches ──────────────────────────────────────────────────
df = assign_batches(df, estimated_batch_size=ESTIMATED_BATCH_SIZE)

# ── Display results ─────────────────────────────────────────────────
print(f"\nEstimated batch size: {ESTIMATED_BATCH_SIZE}")
print(f"Total rows: {len(df)}")
print(f"Number of batches: {df['batch_id'].nunique()}\n")

display_cols = ["row_id", "batch_id", "name", "_normalized", "_phonetic"]
print(df[display_cols].to_string(index=False))

# ── Per-batch summary ───────────────────────────────────────────────
print("\n--- Batch Summary ---")
for bid, group in df.groupby("batch_id"):
    print(f"  Batch {bid}: {len(group)} rows  (rows {group['row_id'].min()}-{group['row_id'].max()})")
