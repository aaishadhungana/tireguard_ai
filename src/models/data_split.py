"""
Train/test splitting for tire failure prediction.

A naive global-timestamp split ("last 20% of days = test") was tried
first and rejected based on the actual data: every failure in the
Milestone 1 dataset occurs between day 10 and day 24 of a 30-day
simulation (verified empirically, not assumed). A time-cutoff split at
day 24 would put close to ZERO failures in the test set, making
precision/recall undefined. This isn't a hypothetical edge case — it's
exactly what happens with this dataset if you split on time.

Instead, we split by TIRE (group split): every row belonging to a given
tire_id goes entirely into train OR entirely into test, never split
across the two. This has two benefits:
1. No leakage — a tire's own history never appears on both sides.
2. Since most tires experience the same small number of failure
   events, a group split keeps the ratio of failure events roughly
   proportional across train/test — though this is no longer exactly
   one-per-tire (see note below).

Note: after the Milestone 1 simulator retune (see tire_simulator.py's
_build_fleet), a tire that fails, gets replaced, and then wears out
again within the simulation window can accumulate more than one
failure event. At default settings this affects a minority of tires.
GroupKFold itself is unaffected either way — it only guarantees a
tire's rows never split across folds, regardless of how many failures
that tire has.

With relatively few tires and failures overall, a SINGLE
train/test split still leaves very few positive examples in whichever
side is smaller. We use GroupKFold cross-validation instead of one
split, reporting mean +/- std across folds, because a lone train/test
split's metrics would have too much variance to trust with this few
positive examples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def tire_level_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single tire-level split, for cases needing one held-out test set
    (e.g. final model evaluation after cross-validation has already
    given a stable performance estimate)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["tire_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def tire_group_kfold(df: pd.DataFrame, n_splits: int = 5):
    """Yields (train_idx, test_idx) for each fold, split by tire_id so
    no tire's rows ever appear in both train and test within a fold."""
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(df, groups=df["tire_id"])


def verify_no_tire_leakage(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    """Returns True iff no tire_id appears in both train and test."""
    train_tires = set(df.iloc[train_idx]["tire_id"])
    test_tires = set(df.iloc[test_idx]["tire_id"])
    return len(train_tires & test_tires) == 0