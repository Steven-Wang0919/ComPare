# -*- coding: utf-8 -*-
"""
fair_tuning.py

Shared fair hyperparameter tuning protocol used by KAN / MLP / GRNN.
"""

from dataclasses import asdict, dataclass
import json

import numpy as np

from common_utils import (
    DEFAULT_STRATIFY_VIEW,
    get_stratify_metadata,
    get_train_val_test_indices,
)


DEFAULT_SELECTION_METRIC = "mean_val_r2"
DEFAULT_TIE_BREAK_METRIC = "mean_val_are"
DEFAULT_BUDGET_PROFILE = "high"
DEFAULT_N_CANDIDATES = 24
DEFAULT_N_REPEATS = 5


@dataclass(frozen=True)
class FairTuningConfig:
    n_candidates: int = DEFAULT_N_CANDIDATES
    n_repeats: int = DEFAULT_N_REPEATS
    selection_metric: str = DEFAULT_SELECTION_METRIC
    tie_break_metric: str = DEFAULT_TIE_BREAK_METRIC
    seed: int = 42
    inner_val_ratio: float = None
    budget_profile: str = DEFAULT_BUDGET_PROFILE


def default_fair_tuning_config(seed=42, inner_val_ratio=None):
    return FairTuningConfig(
        n_candidates=DEFAULT_N_CANDIDATES,
        n_repeats=DEFAULT_N_REPEATS,
        selection_metric=DEFAULT_SELECTION_METRIC,
        tie_break_metric=DEFAULT_TIE_BREAK_METRIC,
        seed=int(seed),
        inner_val_ratio=inner_val_ratio,
        budget_profile=DEFAULT_BUDGET_PROFILE,
    )


def ensure_fair_tuning_config(tuning_config=None, *, seed=42, inner_val_ratio=None):
    if tuning_config is None:
        return default_fair_tuning_config(seed=seed, inner_val_ratio=inner_val_ratio)

    if tuning_config.inner_val_ratio is None:
        return FairTuningConfig(
            n_candidates=int(tuning_config.n_candidates),
            n_repeats=int(tuning_config.n_repeats),
            selection_metric=str(tuning_config.selection_metric),
            tie_break_metric=str(tuning_config.tie_break_metric),
            seed=int(tuning_config.seed),
            inner_val_ratio=inner_val_ratio,
            budget_profile=str(tuning_config.budget_profile),
        )

    return tuning_config


def tuning_config_to_dict(tuning_config):
    return asdict(tuning_config)


def infer_inner_val_ratio(idx_train, idx_val):
    train_count = int(len(idx_train))
    val_count = int(len(idx_val))
    denom = train_count + val_count
    if denom <= 0:
        raise ValueError("Cannot infer inner_val_ratio from empty train/val split.")
    ratio = val_count / float(denom)
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"Invalid inner_val_ratio inferred from outer split: {ratio}")
    return ratio


def _normalize_idx_array(idx, *, name, n_samples=None):
    arr = np.asarray(idx, dtype=int).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if n_samples is not None and (np.any(arr < 0) or np.any(arr >= int(n_samples))):
        raise ValueError(f"{name} contains out-of-bounds indices for n_samples={n_samples}.")
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"{name} contains duplicate indices.")
    return np.sort(arr)


def normalize_inner_splits(inner_splits, *, n_samples=None):
    if inner_splits is None:
        raise ValueError("inner_splits cannot be None.")

    normalized = []
    seen_fold_ids = set()
    for default_fold_id, split in enumerate(list(inner_splits)):
        if not isinstance(split, dict):
            raise TypeError("Each inner split must be a dict.")

        fold_id = int(split.get("fold_id", default_fold_id))
        if fold_id in seen_fold_ids:
            raise ValueError(f"Duplicate inner fold_id detected: {fold_id}")
        seen_fold_ids.add(fold_id)

        idx_train = _normalize_idx_array(
            split.get("idx_train"),
            name=f"inner_splits[{default_fold_id}].idx_train",
            n_samples=n_samples,
        )
        idx_val = _normalize_idx_array(
            split.get("idx_val"),
            name=f"inner_splits[{default_fold_id}].idx_val",
            n_samples=n_samples,
        )
        if len(np.intersect1d(idx_train, idx_val)) > 0:
            raise ValueError(f"inner_splits[{default_fold_id}] has overlapping idx_train and idx_val.")

        split_meta = split.get("split_meta")
        if split_meta is not None and not isinstance(split_meta, dict):
            raise TypeError("split_meta must be a dict when provided.")

        normalized.append({
            "fold_id": fold_id,
            "idx_train": idx_train,
            "idx_val": idx_val,
            "split_meta": dict(split_meta or {}),
        })

    if len(normalized) == 0:
        raise ValueError("inner_splits must contain at least one fold.")
    return sorted(normalized, key=lambda row: int(row["fold_id"]))


def _jsonable_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _rowify_meta(meta):
    row = {}
    for key, value in dict(meta or {}).items():
        if isinstance(value, (list, tuple, dict)):
            row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            row[key] = _jsonable_scalar(value)
    return row


def _normalize_sorted_unique(values):
    return [float(v) for v in sorted(np.unique(np.asarray(values, dtype=float).reshape(-1)).tolist())]


def _speed_contiguity_info(speed_values, reference_speed_values):
    speed_values = _normalize_sorted_unique(speed_values)
    reference_speed_values = _normalize_sorted_unique(reference_speed_values)
    ref_pos = {float(v): idx for idx, v in enumerate(reference_speed_values)}
    if len(speed_values) == 0:
        raise ValueError("speed_values is empty.")
    missing = [float(v) for v in speed_values if float(v) not in ref_pos]
    if missing:
        raise ValueError(f"Speed values {missing} are missing from reference_speed_values.")

    positions = [int(ref_pos[float(v)]) for v in speed_values]
    is_contiguous = all((b - a) == 1 for a, b in zip(positions[:-1], positions[1:]))
    return {
        "speed_values": [float(v) for v in speed_values],
        "speed_min": float(speed_values[0]),
        "speed_max": float(speed_values[-1]),
        "speed_count": int(len(speed_values)),
        "reference_positions": positions,
        "is_contiguous_in_reference": bool(is_contiguous),
    }


def _contiguous_speed_runs(available_speed_values, reference_speed_values):
    available_speed_values = _normalize_sorted_unique(available_speed_values)
    reference_speed_values = _normalize_sorted_unique(reference_speed_values)
    ref_pos = {float(v): idx for idx, v in enumerate(reference_speed_values)}
    positions = [int(ref_pos[float(v)]) for v in available_speed_values]
    runs = []
    current_values = []
    current_positions = []
    for speed_value, pos in zip(available_speed_values, positions):
        if len(current_positions) == 0 or pos == current_positions[-1] + 1:
            current_values.append(float(speed_value))
            current_positions.append(int(pos))
        else:
            runs.append((list(current_values), list(current_positions)))
            current_values = [float(speed_value)]
            current_positions = [int(pos)]
    if current_values:
        runs.append((list(current_values), list(current_positions)))
    return runs


def build_inner_repeated_splits(
    X_dev,
    y_dev,
    tuning_config,
    use_stratify=True,
    *,
    stratify_view=DEFAULT_STRATIFY_VIEW,
):
    splits = []
    for repeat_idx in range(int(tuning_config.n_repeats)):
        idx_train, idx_val, _ = get_train_val_test_indices(
            X=X_dev,
            y=y_dev,
            test_size=0.0,
            val_size=float(tuning_config.inner_val_ratio),
            random_state=int(tuning_config.seed) + repeat_idx,
            use_stratify=use_stratify,
            stratify_view=stratify_view,
        )
        splits.append({
            "fold_id": int(repeat_idx),
            "idx_train": idx_train,
            "idx_val": idx_val,
            "split_meta": {
                "repeat_idx": int(repeat_idx),
            },
        })
    return normalize_inner_splits(splits, n_samples=len(X_dev))


def build_leave_one_opening_out_inner_splits(X_dev, *, opening_col=0, atol=1e-8):
    X_dev = np.asarray(X_dev)
    opening_values = np.asarray(X_dev[:, int(opening_col)], dtype=float).reshape(-1)
    unique_openings = _normalize_sorted_unique(opening_values)
    splits = []
    for fold_id, opening in enumerate(unique_openings):
        val_mask = np.isclose(opening_values, float(opening), atol=atol)
        splits.append({
            "fold_id": int(fold_id),
            "idx_train": np.where(~val_mask)[0],
            "idx_val": np.where(val_mask)[0],
            "split_meta": {
                "inner_holdout_opening": float(opening),
            },
        })
    return normalize_inner_splits(splits, n_samples=len(X_dev))


def build_speed_block_inner_splits(
    X_dev,
    *,
    speed_col=1,
    block_size=5,
    reference_speed_values=None,
):
    X_dev = np.asarray(X_dev)
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")

    speed_values = np.asarray(X_dev[:, int(speed_col)], dtype=float).reshape(-1)
    reference_speed_values = (
        _normalize_sorted_unique(speed_values)
        if reference_speed_values is None
        else _normalize_sorted_unique(reference_speed_values)
    )
    available_speed_values = _normalize_sorted_unique(speed_values)
    speed_runs = _contiguous_speed_runs(available_speed_values, reference_speed_values)

    splits = []
    next_fold_id = 0
    for run_idx, (run_values, run_positions) in enumerate(speed_runs):
        for start in range(0, len(run_values), int(block_size)):
            block_values = run_values[start:start + int(block_size)]
            block_positions = run_positions[start:start + int(block_size)]
            if len(block_values) < int(block_size):
                break
            block_set = {float(v) for v in block_values}
            val_mask = np.array([float(v) in block_set for v in speed_values], dtype=bool)
            contiguity = _speed_contiguity_info(block_values, reference_speed_values)
            splits.append({
                "fold_id": int(next_fold_id),
                "idx_train": np.where(~val_mask)[0],
                "idx_val": np.where(val_mask)[0],
                "split_meta": {
                    "inner_holdout_speed_values": list(contiguity["speed_values"]),
                    "inner_holdout_speed_min": float(contiguity["speed_min"]),
                    "inner_holdout_speed_max": float(contiguity["speed_max"]),
                    "inner_holdout_speed_count": int(contiguity["speed_count"]),
                    "inner_holdout_speed_positions": list(block_positions),
                    "inner_holdout_speed_is_contiguous": bool(contiguity["is_contiguous_in_reference"]),
                    "inner_speed_run_id": int(run_idx),
                },
            })
            next_fold_id += 1

    return normalize_inner_splits(splits, n_samples=len(X_dev))


def build_protocol_aligned_inner_splits(
    X_dev,
    *,
    split_info,
    reference_X=None,
    opening_col=0,
    speed_col=1,
    speed_block_size=5,
):
    protocol = str(split_info.get("protocol", ""))
    if protocol == "leave_one_opening_out":
        inner_splits = build_leave_one_opening_out_inner_splits(
            X_dev,
            opening_col=opening_col,
        )
        return inner_splits, "protocol_aligned_opening_cv", {
            "outer_holdout_opening": float(split_info["holdout_opening"]),
        }

    if protocol == "leave_speed_block_out":
        reference_speed_values = None
        if reference_X is not None:
            reference_speed_values = np.asarray(reference_X)[:, int(speed_col)]
        inner_splits = build_speed_block_inner_splits(
            X_dev,
            speed_col=speed_col,
            block_size=int(speed_block_size),
            reference_speed_values=reference_speed_values,
        )

        ref_speed_arr = (
            np.asarray(reference_X)[:, int(speed_col)]
            if reference_X is not None
            else np.asarray(X_dev)[:, int(speed_col)]
        )
        outer_mask = (
            (ref_speed_arr >= float(split_info["holdout_speed_min"]))
            & (ref_speed_arr <= float(split_info["holdout_speed_max"]))
        )
        outer_speed_values = np.unique(ref_speed_arr[outer_mask]).tolist()
        outer_contiguity = _speed_contiguity_info(
            outer_speed_values,
            reference_speed_values if reference_speed_values is not None else ref_speed_arr,
        )
        return inner_splits, "protocol_aligned_speed_block_cv", {
            "speed_block_size": int(speed_block_size),
            "outer_holdout_speed_values": list(outer_contiguity["speed_values"]),
            "outer_holdout_speed_min": float(outer_contiguity["speed_min"]),
            "outer_holdout_speed_max": float(outer_contiguity["speed_max"]),
            "outer_holdout_speed_is_contiguous": bool(outer_contiguity["is_contiguous_in_reference"]),
        }

    raise ValueError(f"Unsupported protocol for protocol-aligned inner tuning: {protocol}")


def prepare_inner_cv(
    X_dev,
    y_dev,
    tuning_config,
    *,
    inner_splits=None,
    inner_split_strategy=None,
    inner_split_meta=None,
    use_stratify=True,
    stratify_view=DEFAULT_STRATIFY_VIEW,
):
    resolved_meta = {
        **get_stratify_metadata(stratify_view),
        **dict(inner_split_meta or {}),
    }
    if inner_splits is None:
        if tuning_config.inner_val_ratio is None:
            raise ValueError("Repeated-random inner tuning requires tuning_config.inner_val_ratio.")
        resolved_splits = build_inner_repeated_splits(
            X_dev,
            y_dev,
            tuning_config,
            use_stratify=use_stratify,
            stratify_view=stratify_view,
        )
        resolved_strategy = str(inner_split_strategy or "repeated_random")
        resolved_meta.setdefault("inner_val_ratio", float(tuning_config.inner_val_ratio))
        return resolved_splits, resolved_strategy, resolved_meta

    resolved_splits = normalize_inner_splits(inner_splits, n_samples=len(X_dev))
    resolved_strategy = str(inner_split_strategy or "explicit_inner_splits")
    if tuning_config.inner_val_ratio is not None:
        resolved_meta.setdefault("inner_val_ratio", float(tuning_config.inner_val_ratio))
    return resolved_splits, resolved_strategy, resolved_meta


def config_to_json(config):
    return json.dumps(config, ensure_ascii=False, sort_keys=True)


def run_fair_tuning(
    *,
    candidate_configs,
    inner_splits,
    eval_candidate_fn,
    tuning_config,
    model_name,
    task_name,
    inner_split_strategy=None,
    inner_split_meta=None,
):
    if len(candidate_configs) != int(tuning_config.n_candidates):
        raise ValueError(
            f"{model_name}/{task_name} candidate count mismatch: "
            f"expected {tuning_config.n_candidates}, got {len(candidate_configs)}"
        )

    normalized_inner_splits = normalize_inner_splits(inner_splits)
    resolved_inner_split_strategy = str(inner_split_strategy or "unknown")
    resolved_inner_split_meta = dict(inner_split_meta or {})
    inner_fold_count = int(len(normalized_inner_splits))

    tuning_records = []
    candidate_summaries = []

    for candidate_idx, config in enumerate(candidate_configs):
        repeat_results = []
        for split in normalized_inner_splits:
            metrics = eval_candidate_fn(
                config=config,
                idx_train=split["idx_train"],
                idx_val=split["idx_val"],
                fold_id=int(split["fold_id"]),
                split_meta=dict(split.get("split_meta") or {}),
            )
            repeat_results.append(metrics)

        val_r2_values = np.asarray([m["val_r2"] for m in repeat_results], dtype=float)
        val_are_values = np.asarray([m["val_are"] for m in repeat_results], dtype=float)

        mean_val_r2 = float(np.mean(val_r2_values))
        std_val_r2 = float(np.std(val_r2_values, ddof=0))
        mean_val_are = float(np.mean(val_are_values))
        std_val_are = float(np.std(val_are_values, ddof=0))

        candidate_summary = {
            "candidate_idx": int(candidate_idx),
            "config_json": config_to_json(config),
            "mean_val_r2": mean_val_r2,
            "std_val_r2": std_val_r2,
            "mean_val_are": mean_val_are,
            "std_val_are": std_val_are,
            "n_repeats": int(inner_fold_count),
            "inner_fold_count": int(inner_fold_count),
            "inner_split_strategy": resolved_inner_split_strategy,
            "inner_val_ratio": (
                None if tuning_config.inner_val_ratio is None else float(tuning_config.inner_val_ratio)
            ),
            "inner_split_meta_json": config_to_json(resolved_inner_split_meta),
        }
        candidate_summaries.append(candidate_summary)

        for split, metrics in zip(normalized_inner_splits, repeat_results):
            row = {
                "task_name": task_name,
                "model_name": model_name,
                "budget_profile": tuning_config.budget_profile,
                "trial_budget": int(tuning_config.n_candidates),
                "validation_repeats": int(inner_fold_count),
                "selection_metric": tuning_config.selection_metric,
                "tie_break_metric": tuning_config.tie_break_metric,
                "seed": int(tuning_config.seed),
                "inner_val_ratio": (
                    None if tuning_config.inner_val_ratio is None else float(tuning_config.inner_val_ratio)
                ),
                "candidate_idx": int(candidate_idx),
                "inner_fold_id": int(split["fold_id"]),
                "inner_fold_count": int(inner_fold_count),
                "inner_split_strategy": resolved_inner_split_strategy,
                "repeat_train_size": int(len(split["idx_train"])),
                "repeat_val_size": int(len(split["idx_val"])),
                "config_json": config_to_json(config),
                "val_r2": float(metrics["val_r2"]),
                "val_are": float(metrics["val_are"]),
                "mean_val_r2": mean_val_r2,
                "std_val_r2": std_val_r2,
                "mean_val_are": mean_val_are,
                "std_val_are": std_val_are,
                "inner_split_meta_json": config_to_json(resolved_inner_split_meta),
            }
            if resolved_inner_split_strategy == "repeated_random":
                row["repeat_idx"] = int(split["fold_id"])
            row.update(_rowify_meta(split.get("split_meta") or {}))
            tuning_records.append(row)

    best_summary = sorted(
        candidate_summaries,
        key=lambda row: (
            -float(row["mean_val_r2"]),
            float(row["mean_val_are"]),
            float(row["std_val_r2"]),
            int(row["candidate_idx"]),
        ),
    )[0]
    best_candidate_idx = int(best_summary["candidate_idx"])
    best_config = candidate_configs[best_candidate_idx]

    for row in tuning_records:
        row["is_selected"] = int(int(row["candidate_idx"]) == best_candidate_idx)

    return {
        "best_config": best_config,
        "best_candidate_idx": best_candidate_idx,
        "candidate_summaries": candidate_summaries,
        "tuning_records": tuning_records,
        "inner_fold_count": int(inner_fold_count),
        "inner_split_strategy": resolved_inner_split_strategy,
        "inner_split_meta": resolved_inner_split_meta,
    }
