# -*- coding: utf-8 -*-
"""
robustness_utils.py

Shared helpers for multi-replicate experiment management, summary aggregation,
canonical replicate selection, and paired statistical comparison.
"""

from itertools import combinations

import numpy as np
import pandas as pd


DEFAULT_TRAINING_SEEDS = [42, 52, 62, 72, 82]
DEFAULT_OUTER_REPEATS = 5
DEFAULT_SPLIT_SEED_OFFSET = 1000
DEFAULT_STATS_METHOD = "paired_permutation"
DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_CI_METHOD = "paired_bootstrap_percentile"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CANONICAL_RULE = "first_train_seed_first_outer_repeat"
FIXED_HOLDOUT_OUTER_REPEAT_ID = 1
FIXED_HOLDOUT_SPLIT_SEED = -1
PAIR_TIE_TOL = 1e-12


def normalize_training_seeds(training_seeds=None):
    seeds = DEFAULT_TRAINING_SEEDS if training_seeds is None else training_seeds
    if isinstance(seeds, str):
        items = [token.strip() for token in seeds.split(",")]
        seeds = [int(token) for token in items if token]
    normalized = [int(seed) for seed in list(seeds)]
    if len(normalized) == 0:
        raise ValueError("training_seeds must contain at least one seed.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"training_seeds contains duplicates: {normalized}")
    return normalized


def split_seed_for_outer_repeat(outer_repeat_id, *, offset=DEFAULT_SPLIT_SEED_OFFSET):
    return int(offset) + int(outer_repeat_id)


def build_replicate_id(protocol, fold_id, train_seed, outer_repeat_id):
    return (
        f"{protocol}|fold={int(fold_id)}|train_seed={int(train_seed)}|"
        f"outer_repeat_id={int(outer_repeat_id)}"
    )


def build_replicate_record(
    *,
    protocol,
    fold_id,
    train_seed,
    outer_repeat_id,
    split_seed,
    extra=None,
):
    row = {
        "protocol": str(protocol),
        "fold_id": int(fold_id),
        "train_seed": int(train_seed),
        "outer_repeat_id": int(outer_repeat_id),
        "split_seed": int(split_seed),
    }
    row["replicate_id"] = build_replicate_id(
        row["protocol"],
        row["fold_id"],
        row["train_seed"],
        row["outer_repeat_id"],
    )
    row.update(dict(extra or {}))
    return row


def canonical_replicate_rule():
    return DEFAULT_CANONICAL_RULE


def is_canonical_replicate(row, *, primary_seed):
    return (
        int(row["train_seed"]) == int(primary_seed)
        and int(row["outer_repeat_id"]) == FIXED_HOLDOUT_OUTER_REPEAT_ID
    )


def choose_canonical_replicate(df, *, primary_seed):
    if df is None or len(df) == 0:
        raise ValueError("Cannot choose canonical replicate from an empty DataFrame.")
    ordered = df.copy()
    ordered["__is_canonical"] = (
        (ordered["train_seed"].astype(int) == int(primary_seed))
        & (ordered["outer_repeat_id"].astype(int) == FIXED_HOLDOUT_OUTER_REPEAT_ID)
    )
    ordered = ordered.sort_values(
        ["__is_canonical", "train_seed", "outer_repeat_id", "fold_id", "protocol"],
        ascending=[False, True, True, True, True],
    )
    return ordered.iloc[0].drop(labels=["__is_canonical"])


def paired_key_definition_from_cols(pair_key_cols):
    return "(" + ", ".join(str(col) for col in pair_key_cols) + ")"


def summarize_replicate_metrics(
    df,
    *,
    group_cols,
    metric_cols,
    passthrough_cols=None,
    include_min_max=False,
    count_col="n_replicates",
):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    work = df.copy()
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    rows = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row[count_col] = int(len(group))
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[col] = np.nan
                row[f"{col}_std"] = np.nan
                if include_min_max:
                    row[f"{col}_min"] = np.nan
                    row[f"{col}_max"] = np.nan
                continue
            row[col] = float(np.mean(values))
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if include_min_max:
                row[f"{col}_min"] = float(np.min(values))
                row[f"{col}_max"] = float(np.max(values))
        for col in list(passthrough_cols or []):
            row[col] = group.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows)


def percentile_bootstrap_ci(
    delta,
    *,
    alpha=0.05,
    n_boot=DEFAULT_BOOTSTRAP_SAMPLES,
    seed=0,
):
    values = np.asarray(delta, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        single = float(values[0])
        return single, single
    rng = np.random.default_rng(int(seed))
    sample_idx = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    means = values[sample_idx].mean(axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def paired_permutation_pvalue(
    delta,
    *,
    n_permutations=DEFAULT_N_PERMUTATIONS,
    seed=0,
):
    values = np.asarray(delta, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, "paired_permutation", 0
    observed = float(np.mean(values))
    if n == 1:
        return 1.0, "paired_permutation_exact", 2
    if n <= 16:
        masks = np.arange(1 << n, dtype=np.uint32)[:, None]
        bit_pos = np.arange(n, dtype=np.uint32)[None, :]
        bits = ((masks >> bit_pos) & 1).astype(np.int8)
        signs = 1 - 2 * bits
        permuted = (signs * values[None, :]).mean(axis=1)
        p_value = float(np.mean(np.abs(permuted) >= abs(observed) - PAIR_TIE_TOL))
        return p_value, "paired_permutation_exact", int(1 << n)
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_permutations), n))
    permuted = (signs * values[None, :]).mean(axis=1)
    p_value = float(np.mean(np.abs(permuted) >= abs(observed) - PAIR_TIE_TOL))
    return p_value, "paired_permutation_monte_carlo", int(n_permutations)


def build_pairwise_stats(
    df,
    *,
    metric_specs,
    model_col,
    pair_key_cols,
    analysis_group_cols=None,
    stats_seed=0,
):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    analysis_group_cols = list(analysis_group_cols or [])
    pair_key_cols = list(pair_key_cols)
    pair_key_def = paired_key_definition_from_cols(pair_key_cols)
    rows = []
    models = [str(model) for model in sorted(df[model_col].astype(str).unique().tolist())]
    work = df.copy()
    for spec in metric_specs:
        work[spec["column"]] = pd.to_numeric(work[spec["column"]], errors="coerce")

    grouped = (
        [((), work)]
        if len(analysis_group_cols) == 0
        else list(work.groupby(analysis_group_cols, dropna=False, sort=False))
    )
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        analysis_meta = {col: value for col, value in zip(analysis_group_cols, keys)}
        for spec in metric_specs:
            metric_col = spec["column"]
            higher_is_better = bool(spec.get("higher_is_better", True))
            subset = group[pair_key_cols + [model_col, metric_col]].dropna(subset=[metric_col])
            if len(subset) == 0:
                continue
            pivot = subset.pivot_table(
                index=pair_key_cols,
                columns=model_col,
                values=metric_col,
                aggfunc="first",
            )
            available_models = [model for model in models if model in pivot.columns]
            for model_a, model_b in combinations(available_models, 2):
                aligned = pivot[[model_a, model_b]].dropna()
                if len(aligned) == 0:
                    continue
                raw_delta = aligned[model_a].to_numpy(dtype=float) - aligned[model_b].to_numpy(dtype=float)
                delta = raw_delta if higher_is_better else (-1.0 * raw_delta)
                ci_low, ci_high = percentile_bootstrap_ci(
                    delta,
                    seed=int(stats_seed) + len(rows),
                )
                p_value, stats_method, permutations_used = paired_permutation_pvalue(
                    delta,
                    seed=int(stats_seed) + len(rows),
                )
                wins = int(np.sum(delta > PAIR_TIE_TOL))
                losses = int(np.sum(delta < -PAIR_TIE_TOL))
                ties = int(len(delta) - wins - losses)
                mean_delta = float(np.mean(delta))
                std_delta = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
                if mean_delta > PAIR_TIE_TOL:
                    better_model = model_a
                elif mean_delta < -PAIR_TIE_TOL:
                    better_model = model_b
                else:
                    better_model = "tie"
                row = dict(analysis_meta)
                row.update({
                    "metric": metric_col,
                    "model_a": model_a,
                    "model_b": model_b,
                    "paired_n": int(len(delta)),
                    "mean_delta": mean_delta,
                    "std_delta": std_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "win_count": wins,
                    "tie_count": ties,
                    "loss_count": losses,
                    "p_value": p_value,
                    "stats_method": stats_method,
                    "n_permutations": permutations_used,
                    "ci_method": DEFAULT_CI_METHOD,
                    "paired_key_definition": pair_key_def,
                    "higher_is_better": int(higher_is_better),
                    "better_model_by_mean": better_model,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def build_protocol_summary_wide(df, *, protocol_col="protocol", model_col="model"):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    summary = summarize_replicate_metrics(
        df,
        group_cols=[protocol_col, model_col],
        metric_cols=["r2", "are"],
        include_min_max=False,
    )
    if len(summary) == 0:
        return pd.DataFrame()

    rows = []
    for protocol, group in summary.groupby(protocol_col, sort=False):
        row = {protocol_col: protocol}
        for _, item in group.iterrows():
            model_name = str(item[model_col])
            row[f"r2_{model_name}"] = item["r2"]
            row[f"r2_{model_name}_std"] = item["r2_std"]
            row[f"are_{model_name}"] = item["are"]
            row[f"are_{model_name}_std"] = item["are_std"]
            row[f"n_replicates_{model_name}"] = item["n_replicates"]
        rows.append(row)
    return pd.DataFrame(rows)
