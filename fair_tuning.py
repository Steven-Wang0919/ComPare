# -*- coding: utf-8 -*-
"""
fair_tuning.py

Shared fair hyperparameter tuning protocol used by KAN / MLP / GRNN.
"""

from dataclasses import asdict, dataclass
import json

import numpy as np

from common_utils import get_train_val_test_indices


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


def build_inner_repeated_splits(X_dev, y_dev, tuning_config, use_stratify=True):
    splits = []
    for repeat_idx in range(int(tuning_config.n_repeats)):
        idx_train, idx_val, _ = get_train_val_test_indices(
            X=X_dev,
            y=y_dev,
            test_size=0.0,
            val_size=float(tuning_config.inner_val_ratio),
            random_state=int(tuning_config.seed) + repeat_idx,
            use_stratify=use_stratify,
        )
        splits.append({
            "repeat_idx": repeat_idx,
            "idx_train": idx_train,
            "idx_val": idx_val,
        })
    return splits


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
):
    if len(candidate_configs) != int(tuning_config.n_candidates):
        raise ValueError(
            f"{model_name}/{task_name} candidate count mismatch: "
            f"expected {tuning_config.n_candidates}, got {len(candidate_configs)}"
        )

    tuning_records = []
    candidate_summaries = []

    for candidate_idx, config in enumerate(candidate_configs):
        repeat_results = []
        for split in inner_splits:
            metrics = eval_candidate_fn(
                config=config,
                idx_train=split["idx_train"],
                idx_val=split["idx_val"],
                repeat_idx=split["repeat_idx"],
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
            "n_repeats": int(tuning_config.n_repeats),
        }
        candidate_summaries.append(candidate_summary)

        for split, metrics in zip(inner_splits, repeat_results):
            tuning_records.append({
                "task_name": task_name,
                "model_name": model_name,
                "budget_profile": tuning_config.budget_profile,
                "trial_budget": int(tuning_config.n_candidates),
                "validation_repeats": int(tuning_config.n_repeats),
                "selection_metric": tuning_config.selection_metric,
                "tie_break_metric": tuning_config.tie_break_metric,
                "seed": int(tuning_config.seed),
                "inner_val_ratio": float(tuning_config.inner_val_ratio),
                "candidate_idx": int(candidate_idx),
                "repeat_idx": int(split["repeat_idx"]),
                "repeat_train_size": int(len(split["idx_train"])),
                "repeat_val_size": int(len(split["idx_val"])),
                "config_json": config_to_json(config),
                "val_r2": float(metrics["val_r2"]),
                "val_are": float(metrics["val_are"]),
                "mean_val_r2": mean_val_r2,
                "std_val_r2": std_val_r2,
                "mean_val_are": mean_val_are,
                "std_val_are": std_val_are,
            })

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
    }
