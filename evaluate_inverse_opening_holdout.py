# -*- coding: utf-8 -*-
"""
evaluate_inverse_opening_holdout.py

Inverse leave-one-opening-out evaluation with robustness summaries.
"""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

from common_utils import build_protocol_splits, build_sample_tracking_columns, load_data_with_metadata
from inverse_grnn import train_and_eval_inverse_grnn
from inverse_kan import train_and_eval_inverse_kan_v2
from inverse_mlp import train_and_eval_inverse_mlp
from policy_config import (
    POLICY_LABEL,
    POLICY_LOW_MID_THRESHOLD,
    POLICY_MID_HIGH_THRESHOLD,
    POLICY_TARGET_OPENINGS,
)
from robustness_utils import (
    DEFAULT_OUTER_REPEATS,
    DEFAULT_STATS_METHOD,
    FIXED_HOLDOUT_OUTER_REPEAT_ID,
    FIXED_HOLDOUT_SPLIT_SEED,
    build_pairwise_stats,
    build_replicate_record,
    canonical_replicate_rule,
    is_canonical_replicate,
    normalize_training_seeds,
    summarize_replicate_metrics,
)
from run_utils import (
    append_manifest_outputs,
    build_multi_fold_split_artifact_payload,
    create_run_dir,
    save_dataframe,
    write_manifest,
)


VAL_RATIO = 0.2
CANONICAL_RULE = canonical_replicate_rule()


def _artifact_source_files():
    base_dir = os.path.dirname(__file__)
    return [
        __file__,
        os.path.join(base_dir, "inverse_mlp.py"),
        os.path.join(base_dir, "inverse_grnn.py"),
        os.path.join(base_dir, "inverse_kan.py"),
        os.path.join(base_dir, "common_utils.py"),
        os.path.join(base_dir, "fair_tuning.py"),
        os.path.join(base_dir, "policy_config.py"),
        os.path.join(base_dir, "robustness_utils.py"),
        os.path.join(base_dir, "run_utils.py"),
    ]


def _artifact_outputs_from_result(result, run_dir):
    outputs = []
    for key in [
        "artifact_model_path",
        "artifact_meta_path",
        "artifact_test_inputs_path",
        "artifact_test_targets_path",
    ]:
        path = result.get(key)
        if path:
            outputs.append({"path": os.path.relpath(path, run_dir).replace("\\", "/")})
    return outputs


def _cleanup_runtime():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception:
        pass


def _to_1d_array(arr, name):
    out = np.asarray(arr).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} is empty.")
    return out


def _validate_same_length(arr1, arr2, name1, name2):
    if len(arr1) != len(arr2):
        raise ValueError(f"{name1} and {name2} have different lengths.")


def _validate_same_values(arr1, arr2, name1, name2, atol=1e-8, rtol=1e-6):
    a1 = np.asarray(arr1).reshape(-1)
    a2 = np.asarray(arr2).reshape(-1)
    _validate_same_length(a1, a2, name1, name2)
    if not np.allclose(a1, a2, atol=atol, rtol=rtol, equal_nan=False):
        bad = np.where(~np.isclose(a1, a2, atol=atol, rtol=rtol, equal_nan=False))[0]
        idx = int(bad[0])
        raise ValueError(
            f"{name1} and {name2} differ at index {idx}: "
            f"{name1}={a1[idx]}, {name2}={a2[idx]}"
        )


def _validate_same_mask(mask1, mask2, name1, name2):
    m1 = np.asarray(mask1).astype(bool).reshape(-1)
    m2 = np.asarray(mask2).astype(bool).reshape(-1)
    _validate_same_length(m1, m2, name1, name2)
    if not np.array_equal(m1, m2):
        bad = np.where(m1 != m2)[0]
        idx = int(bad[0])
        raise ValueError(
            f"{name1} and {name2} differ at index {idx}: "
            f"{name1}={m1[idx]}, {name2}={m2[idx]}"
        )


def _fmt_float(x, ndigits=6):
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NaN"
        return f"{float(x):.{ndigits}g}"
    return str(x)


def _format_openings(openings):
    return [float(op) for op in sorted(np.asarray(openings, dtype=float).tolist())]


def _format_openings_str(openings):
    formatted = []
    for op in _format_openings(openings):
        if float(op).is_integer():
            formatted.append(str(int(op)))
        else:
            formatted.append(_fmt_float(op))
    return ",".join(formatted)


def _append_replicate_meta(df, replicate):
    out = df.copy()
    out["protocol"] = replicate["protocol"]
    out["fold_id"] = int(replicate["fold_id"])
    out["train_seed"] = int(replicate["train_seed"])
    out["outer_repeat_id"] = int(replicate["outer_repeat_id"])
    out["split_seed"] = int(replicate["split_seed"])
    out["replicate_id"] = replicate["replicate_id"]
    out["is_canonical_replicate"] = 1 if replicate["is_canonical_replicate"] else 0
    out["canonical_replicate_rule"] = CANONICAL_RULE
    return out


def _build_opening_folds(X, y, train_seed):
    unique_openings = _format_openings(np.unique(np.asarray(X)[:, 0]))
    folds = []
    for fold_idx, opening in enumerate(unique_openings, start=1):
        split_info = build_protocol_splits(
            X,
            y,
            protocol="leave_one_opening_out",
            random_state=int(train_seed),
            val_size=VAL_RATIO,
            holdout_opening=opening,
        )
        train_val_openings = [op for op in unique_openings if not np.isclose(op, opening, atol=1e-8)]
        folds.append({
            "fold_id": fold_idx,
            "test_opening_mm": float(opening),
            "train_val_openings_mm": train_val_openings,
            "train_val_openings_label": _format_openings_str(train_val_openings),
            "train_size": int(len(split_info["idx_train"])),
            "val_size": int(len(split_info["idx_val"])),
            "test_size": int(len(split_info["idx_test"])),
            "split_info": split_info,
        })
    return unique_openings, folds


def _collect_fold_outputs(fold_info, model_results, sample_meta, replicate):
    mlp_res = model_results["inverse_MLP"]
    grnn_res = model_results["inverse_GRNN"]
    kan_res = model_results["inverse_KAN"]

    y_true_all_mlp = _to_1d_array(mlp_res["y_true_all"], "inverse_MLP y_true_all")
    y_true_all_grnn = _to_1d_array(grnn_res["y_true_all"], "inverse_GRNN y_true_all")
    y_true_all_kan = _to_1d_array(kan_res["y_true_all"], "inverse_KAN y_true_all")
    y_pred_all_mlp = _to_1d_array(mlp_res["y_pred_all"], "inverse_MLP y_pred_all")
    y_pred_all_grnn = _to_1d_array(grnn_res["y_pred_all"], "inverse_GRNN y_pred_all")
    y_pred_all_kan = _to_1d_array(kan_res["y_pred_all"], "inverse_KAN y_pred_all")
    opening_all_mlp = _to_1d_array(mlp_res["opening_all"], "inverse_MLP opening_all")
    opening_all_grnn = _to_1d_array(grnn_res["opening_all"], "inverse_GRNN opening_all")
    opening_all_kan = _to_1d_array(kan_res["opening_all"], "inverse_KAN opening_all")
    mass_all_mlp = _to_1d_array(mlp_res["mass_all"], "inverse_MLP mass_all")
    mass_all_grnn = _to_1d_array(grnn_res["mass_all"], "inverse_GRNN mass_all")
    mass_all_kan = _to_1d_array(kan_res["mass_all"], "inverse_KAN mass_all")
    strat_open_all_mlp = _to_1d_array(mlp_res["strategy_opening_all"], "inverse_MLP strategy_opening_all")
    strat_open_all_grnn = _to_1d_array(grnn_res["strategy_opening_all"], "inverse_GRNN strategy_opening_all")
    strat_open_all_kan = _to_1d_array(kan_res["strategy_opening_all"], "inverse_KAN strategy_opening_all")
    policy_mask_mlp = np.asarray(mlp_res["policy_mask"]).astype(bool).reshape(-1)
    policy_mask_grnn = np.asarray(grnn_res["policy_mask"]).astype(bool).reshape(-1)
    policy_mask_kan = np.asarray(kan_res["policy_mask"]).astype(bool).reshape(-1)

    _validate_same_values(y_true_all_mlp, y_true_all_grnn, "inverse_MLP y_true_all", "inverse_GRNN y_true_all")
    _validate_same_values(y_true_all_mlp, y_true_all_kan, "inverse_MLP y_true_all", "inverse_KAN y_true_all")
    _validate_same_values(mass_all_mlp, mass_all_grnn, "inverse_MLP mass_all", "inverse_GRNN mass_all")
    _validate_same_values(mass_all_mlp, mass_all_kan, "inverse_MLP mass_all", "inverse_KAN mass_all")
    _validate_same_values(opening_all_mlp, opening_all_grnn, "inverse_MLP opening_all", "inverse_GRNN opening_all")
    _validate_same_values(opening_all_mlp, opening_all_kan, "inverse_MLP opening_all", "inverse_KAN opening_all")
    _validate_same_values(strat_open_all_mlp, strat_open_all_grnn, "inverse_MLP strategy_opening_all", "inverse_GRNN strategy_opening_all")
    _validate_same_values(strat_open_all_mlp, strat_open_all_kan, "inverse_MLP strategy_opening_all", "inverse_KAN strategy_opening_all")
    _validate_same_mask(policy_mask_mlp, policy_mask_grnn, "inverse_MLP policy_mask", "inverse_GRNN policy_mask")
    _validate_same_mask(policy_mask_mlp, policy_mask_kan, "inverse_MLP policy_mask", "inverse_KAN policy_mask")

    test_indices = np.asarray(fold_info["split_info"]["idx_test"], dtype=int).reshape(-1)
    common_cols = {
        "fold_id": int(fold_info["fold_id"]),
        "test_opening_mm": float(fold_info["test_opening_mm"]),
        "train_val_openings_mm": fold_info["train_val_openings_label"],
        **build_sample_tracking_columns(sample_meta, test_indices, include_legacy_sample_index=True),
        "target_mass_g_min": mass_all_mlp,
        "actual_opening_mm": opening_all_mlp,
        "strategy_opening_mm": strat_open_all_mlp,
        "true_speed_r_min": y_true_all_mlp,
    }
    df_all = pd.DataFrame({
        **common_cols,
        "inverse_MLP_pred": y_pred_all_mlp,
        "inverse_GRNN_pred": y_pred_all_grnn,
        "inverse_KAN_pred": y_pred_all_kan,
        "policy_match": policy_mask_mlp.astype(int),
    })

    main_idx = np.where(policy_mask_mlp)[0]
    df_main = pd.DataFrame({
        "fold_id": int(fold_info["fold_id"]),
        "test_opening_mm": float(fold_info["test_opening_mm"]),
        "train_val_openings_mm": fold_info["train_val_openings_label"],
        **build_sample_tracking_columns(sample_meta, test_indices[main_idx], include_legacy_sample_index=True),
        "target_mass_g_min": mass_all_mlp[main_idx],
        "actual_opening_mm": opening_all_mlp[main_idx],
        "strategy_opening_mm": strat_open_all_mlp[main_idx],
        "true_speed_r_min": y_true_all_mlp[main_idx],
        "inverse_MLP_pred": y_pred_all_mlp[main_idx],
        "inverse_GRNN_pred": y_pred_all_grnn[main_idx],
        "inverse_KAN_pred": y_pred_all_kan[main_idx],
    })

    return _append_replicate_meta(df_all, replicate), _append_replicate_meta(df_main, replicate)


def _format_hyperparams(model_name, result):
    if model_name == "inverse_MLP":
        return f"hidden={result.get('best_hidden')}, alpha={_fmt_float(result.get('best_alpha'))}"
    if model_name == "inverse_GRNN":
        return f"sigma={_fmt_float(result.get('best_sigma'))}"
    return (
        f"hidden={result.get('best_hidden_dim')}, "
        f"lr={_fmt_float(result.get('best_lr'))}, "
        f"wd={_fmt_float(result.get('best_weight_decay'))}"
    )


def _merge_summary_hyperparams(df_summary, df_metrics):
    if len(df_summary) == 0:
        return df_summary
    base = (
        df_metrics.sort_values(
            ["is_canonical_replicate", "train_seed", "fold_id"],
            ascending=[False, True, True],
        )[["Model", "Hyperparams"]]
        .drop_duplicates(subset=["Model"], keep="first")
    )
    return df_summary.merge(base, on="Model", how="left")


def _build_replicate_jobs(X_raw, y_raw, training_seeds, primary_seed):
    jobs = []
    split_payload_folds = []
    payload_fold_id = 1
    unique_openings = None
    for train_seed in training_seeds:
        unique_openings, folds = _build_opening_folds(X_raw, y_raw, train_seed)
        for fold_info in folds:
            protocol_name = f"leave_opening_{_fmt_float(fold_info['test_opening_mm'])}_out"
            replicate = build_replicate_record(
                protocol=protocol_name,
                fold_id=int(fold_info["fold_id"]),
                train_seed=int(train_seed),
                outer_repeat_id=FIXED_HOLDOUT_OUTER_REPEAT_ID,
                split_seed=FIXED_HOLDOUT_SPLIT_SEED,
            )
            replicate["is_canonical_replicate"] = bool(
                is_canonical_replicate(replicate, primary_seed=primary_seed)
            )
            jobs.append((replicate, fold_info))
            split_payload_folds.append({
                "fold_id": payload_fold_id,
                "protocol": protocol_name,
                "idx_train": fold_info["split_info"]["idx_train"],
                "idx_val": fold_info["split_info"]["idx_val"],
                "idx_test": fold_info["split_info"]["idx_test"],
                "test_opening_mm": float(fold_info["test_opening_mm"]),
                "train_seed": replicate["train_seed"],
                "outer_repeat_id": replicate["outer_repeat_id"],
                "split_seed": replicate["split_seed"],
                "replicate_id": replicate["replicate_id"],
            })
            payload_fold_id += 1
    return unique_openings, jobs, split_payload_folds


def run_inverse_opening_holdout_compare(
    output_dir,
    data_path="data/dataset.xlsx",
    training_seeds=None,
    *,
    seed=None,
    X_raw=None,
    y_raw=None,
    sample_meta=None,
    folds=None,
):
    if X_raw is None or y_raw is None or sample_meta is None:
        X_raw, y_raw, sample_meta = load_data_with_metadata(data_path)

    if training_seeds is None and seed is not None:
        training_seeds = [int(seed)]
    training_seeds = normalize_training_seeds(training_seeds)
    primary_seed = int(training_seeds[0])
    if folds is None:
        unique_openings, jobs, _ = _build_replicate_jobs(X_raw, y_raw, training_seeds, primary_seed)
    else:
        unique_openings = _format_openings(np.unique(np.asarray(X_raw)[:, 0]))
        jobs = []
        for train_seed in training_seeds:
            for fold_info in folds:
                protocol_name = f"leave_opening_{_fmt_float(fold_info['test_opening_mm'])}_out"
                replicate = build_replicate_record(
                    protocol=protocol_name,
                    fold_id=int(fold_info["fold_id"]),
                    train_seed=int(train_seed),
                    outer_repeat_id=FIXED_HOLDOUT_OUTER_REPEAT_ID,
                    split_seed=FIXED_HOLDOUT_SPLIT_SEED,
                )
                replicate["is_canonical_replicate"] = bool(
                    is_canonical_replicate(replicate, primary_seed=primary_seed)
                )
                jobs.append((replicate, fold_info))

    metrics_rows = []
    predictions_all = []
    predictions_main = []
    tuning_rows = []
    artifact_outputs = []

    for replicate, fold_info in jobs:
        split_info = fold_info["split_info"]
        split_indices = (
            split_info["idx_train"],
            split_info["idx_val"],
            split_info["idx_test"],
        )
        save_artifacts = bool(replicate["is_canonical_replicate"])
        pred_all_relpath = "inverse_opening_holdout_predictions_all.csv"
        artifact_root = os.path.join(
            output_dir,
            "artifacts",
            f"fold_{int(fold_info['fold_id']):02d}_opening_{_fmt_float(fold_info['test_opening_mm'])}",
        )

        mlp_res = train_and_eval_inverse_mlp(
            data_path=data_path,
            random_state=int(replicate["train_seed"]),
            save_outputs_dir=None,
            split_indices=split_indices,
            save_artifacts=save_artifacts,
            artifact_dir=os.path.join(artifact_root, "inverse_MLP") if save_artifacts else None,
            save_test_slice=save_artifacts,
            artifact_extra={
                "run_dir": output_dir.replace("\\", "/"),
                "fold_id": int(fold_info["fold_id"]),
                "test_opening_mm": float(fold_info["test_opening_mm"]),
                "train_val_openings_mm": fold_info["train_val_openings_label"],
                "train_seed": int(replicate["train_seed"]),
                "outer_repeat_id": int(replicate["outer_repeat_id"]),
                "split_seed": int(replicate["split_seed"]),
                "replicate_id": replicate["replicate_id"],
                "is_canonical_replicate": int(replicate["is_canonical_replicate"]),
                "canonical_replicate_rule": CANONICAL_RULE,
                "reference_output": {
                    "path": pred_all_relpath,
                    "filter_column": "fold_id",
                    "filter_value": int(fold_info["fold_id"]),
                    "prediction_column": "inverse_MLP_pred",
                    "target_column": "true_speed_r_min",
                },
            },
        )
        grnn_res = train_and_eval_inverse_grnn(
            data_path=data_path,
            save_outputs_dir=None,
            split_indices=split_indices,
            random_state=int(replicate["train_seed"]),
            save_artifacts=save_artifacts,
            artifact_dir=os.path.join(artifact_root, "inverse_GRNN") if save_artifacts else None,
            save_test_slice=save_artifacts,
            artifact_extra={
                "run_dir": output_dir.replace("\\", "/"),
                "fold_id": int(fold_info["fold_id"]),
                "test_opening_mm": float(fold_info["test_opening_mm"]),
                "train_val_openings_mm": fold_info["train_val_openings_label"],
                "train_seed": int(replicate["train_seed"]),
                "outer_repeat_id": int(replicate["outer_repeat_id"]),
                "split_seed": int(replicate["split_seed"]),
                "replicate_id": replicate["replicate_id"],
                "is_canonical_replicate": int(replicate["is_canonical_replicate"]),
                "canonical_replicate_rule": CANONICAL_RULE,
                "reference_output": {
                    "path": pred_all_relpath,
                    "filter_column": "fold_id",
                    "filter_value": int(fold_info["fold_id"]),
                    "prediction_column": "inverse_GRNN_pred",
                    "target_column": "true_speed_r_min",
                },
            },
        )
        kan_res = train_and_eval_inverse_kan_v2(
            data_path=data_path,
            seed=int(replicate["train_seed"]),
            save_outputs_dir=None,
            split_indices=split_indices,
            save_artifacts=save_artifacts,
            artifact_dir=os.path.join(artifact_root, "inverse_KAN") if save_artifacts else None,
            save_test_slice=save_artifacts,
            artifact_extra={
                "run_dir": output_dir.replace("\\", "/"),
                "fold_id": int(fold_info["fold_id"]),
                "test_opening_mm": float(fold_info["test_opening_mm"]),
                "train_val_openings_mm": fold_info["train_val_openings_label"],
                "train_seed": int(replicate["train_seed"]),
                "outer_repeat_id": int(replicate["outer_repeat_id"]),
                "split_seed": int(replicate["split_seed"]),
                "replicate_id": replicate["replicate_id"],
                "is_canonical_replicate": int(replicate["is_canonical_replicate"]),
                "canonical_replicate_rule": CANONICAL_RULE,
                "reference_output": {
                    "path": pred_all_relpath,
                    "filter_column": "fold_id",
                    "filter_value": int(fold_info["fold_id"]),
                    "prediction_column": "inverse_KAN_pred",
                    "target_column": "true_speed_r_min",
                },
            },
        )

        fold_results = {
            "inverse_MLP": mlp_res,
            "inverse_GRNN": grnn_res,
            "inverse_KAN": kan_res,
        }
        for model_name, result in fold_results.items():
            metrics_rows.append({
                "Task": "inverse_opening_holdout",
                "protocol": replicate["protocol"],
                "fold_id": int(fold_info["fold_id"]),
                "train_seed": int(replicate["train_seed"]),
                "outer_repeat_id": int(replicate["outer_repeat_id"]),
                "split_seed": int(replicate["split_seed"]),
                "replicate_id": replicate["replicate_id"],
                "is_canonical_replicate": int(replicate["is_canonical_replicate"]),
                "canonical_replicate_rule": CANONICAL_RULE,
                "test_opening_mm": float(fold_info["test_opening_mm"]),
                "train_val_openings_mm": fold_info["train_val_openings_label"],
                "train_size": int(fold_info["train_size"]),
                "val_size": int(fold_info["val_size"]),
                "test_size": int(fold_info["test_size"]),
                "n_test": int(fold_info["test_size"]),
                "Model": model_name,
                "DisplayName": model_name,
                "ArchitectureNote": f"leave-one-opening-out; test opening = {fold_info['test_opening_mm']:g} mm",
                "R2_main": result["r2_main"],
                "ARE_main(%)": result["are_main"],
                "n_main": result["n_main"],
                "main_ratio": result["main_ratio"],
                "R2_all": result["r2_all"],
                "ARE_all(%)": result["are_all"],
                "n_all": result["n_all"],
                "Hyperparams": _format_hyperparams(model_name, result),
            })
            for row in result.get("tuning_records", []):
                merged = {
                    "evaluation_scope": "inverse_opening_holdout",
                    "protocol": replicate["protocol"],
                    "fold_id": int(fold_info["fold_id"]),
                    "train_seed": int(replicate["train_seed"]),
                    "outer_repeat_id": int(replicate["outer_repeat_id"]),
                    "split_seed": int(replicate["split_seed"]),
                    "replicate_id": replicate["replicate_id"],
                    "test_opening_mm": float(fold_info["test_opening_mm"]),
                    "train_val_openings_mm": fold_info["train_val_openings_label"],
                    "reported_model": model_name,
                }
                merged.update(row)
                tuning_rows.append(merged)

        df_fold_all, df_fold_main = _collect_fold_outputs(fold_info, fold_results, sample_meta, replicate)
        if replicate["is_canonical_replicate"]:
            predictions_all.append(df_fold_all)
            predictions_main.append(df_fold_main)
        if save_artifacts:
            for result in fold_results.values():
                artifact_outputs.extend(_artifact_outputs_from_result(result, output_dir))

    df_metrics = pd.DataFrame(metrics_rows)
    df_summary = summarize_replicate_metrics(
        df_metrics,
        group_cols=["Task", "Model", "DisplayName"],
        metric_cols=["R2_main", "ARE_main(%)", "R2_all", "ARE_all(%)", "n_main", "main_ratio", "n_all"],
        include_min_max=True,
    )
    df_summary = _merge_summary_hyperparams(df_summary, df_metrics).sort_values("Model").reset_index(drop=True)
    df_pairwise = build_pairwise_stats(
        df_metrics,
        metric_specs=[
            {"column": "R2_main", "higher_is_better": True},
            {"column": "ARE_main(%)", "higher_is_better": False},
            {"column": "R2_all", "higher_is_better": True},
            {"column": "ARE_all(%)", "higher_is_better": False},
        ],
        model_col="Model",
        pair_key_cols=["protocol", "fold_id", "train_seed"],
        stats_seed=int(primary_seed),
    )
    df_all = pd.concat(predictions_all, ignore_index=True) if len(predictions_all) > 0 else pd.DataFrame()
    df_main = pd.concat(predictions_main, ignore_index=True) if len(predictions_main) > 0 else pd.DataFrame()

    metrics_path = os.path.join(output_dir, "inverse_opening_holdout_metrics.csv")
    summary_path = os.path.join(output_dir, "inverse_opening_holdout_summary.csv")
    pairwise_path = os.path.join(output_dir, "inverse_opening_holdout_pairwise_stats.csv")
    all_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_all.csv")
    main_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_main.csv")

    save_dataframe(df_metrics, metrics_path)
    save_dataframe(df_summary, summary_path)
    save_dataframe(df_pairwise, pairwise_path)
    save_dataframe(df_all, all_path)
    save_dataframe(df_main, main_path)

    tuning_audit_path = None
    if len(tuning_rows) > 0:
        tuning_audit_path = os.path.join(output_dir, "inverse_opening_holdout_tuning_audit.csv")
        save_dataframe(pd.DataFrame(tuning_rows), tuning_audit_path)

    return {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "all_path": all_path,
        "main_path": main_path,
        "tuning_audit_path": tuning_audit_path,
        "artifact_outputs": artifact_outputs,
        "unique_openings": unique_openings,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inverse opening holdout robustness.")
    parser.add_argument("--data-path", default="data/dataset.xlsx")
    parser.add_argument("--seeds", default="42,52,62,72,82", help="Comma-separated training seeds.")
    parser.add_argument("--outer-repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    parser.add_argument("--stats-method", default=DEFAULT_STATS_METHOD)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    training_seeds = normalize_training_seeds(args.seeds)
    primary_seed = int(training_seeds[0])
    X_raw, y_raw, sample_meta = load_data_with_metadata(data_path)
    unique_openings, _, split_payload_folds = _build_replicate_jobs(X_raw, y_raw, training_seeds, primary_seed)
    split_payload = build_multi_fold_split_artifact_payload(
        split_payload_folds,
        n_samples=len(X_raw),
    )
    run_dir = create_run_dir("evaluate_inverse_opening_holdout")

    manifest_path = write_manifest(
        run_dir,
        script_name="evaluate_inverse_opening_holdout.py",
        data_path=data_path,
        seed=training_seeds,
        params={
            "protocol": "leave_one_opening_out_cv",
            "fold_openings_mm": unique_openings,
            "n_folds": len(unique_openings),
            "val_ratio": VAL_RATIO,
            "primary_seed": primary_seed,
            "training_seeds": training_seeds,
            "outer_repeats": int(args.outer_repeats),
            "stats_method": args.stats_method,
            "split_seed_policy": "fixed holdout; no real outer split; split_seed = -1 placeholder",
            "paired_key_definition": "(protocol, fold_id, train_seed)",
            "canonical_replicate_rule": CANONICAL_RULE,
            "policy": {
                "label": POLICY_LABEL,
                "target_openings_mm": list(POLICY_TARGET_OPENINGS),
                "threshold_low_mid": POLICY_LOW_MID_THRESHOLD,
                "threshold_mid_high": POLICY_MID_HIGH_THRESHOLD,
            },
            "note": "each opening fold is evaluated across matched train_seed replicates only",
            "fair_tuning": {
                "n_candidates": 24,
                "n_repeats": 5,
                "selection_metric": "mean_val_r2",
                "tie_break_metric": "mean_val_are",
                "budget_profile": "high",
            },
        },
        source_files=_artifact_source_files(),
        split_payload=split_payload,
    )

    print(f"\nRun directory: {run_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    outputs = run_inverse_opening_holdout_compare(
        run_dir,
        data_path=data_path,
        training_seeds=training_seeds,
        X_raw=X_raw,
        y_raw=y_raw,
        sample_meta=sample_meta,
    )

    manifest_outputs = [
        {"path": os.path.relpath(outputs["metrics_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["summary_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["pairwise_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["all_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["main_path"], run_dir).replace("\\", "/")},
    ]
    if outputs["tuning_audit_path"] is not None:
        manifest_outputs.append({"path": os.path.relpath(outputs["tuning_audit_path"], run_dir).replace("\\", "/")})
    manifest_outputs.extend(outputs["artifact_outputs"])

    append_manifest_outputs(run_dir, manifest_outputs)

    print("\nAll outputs saved to:", flush=True)
    print(run_dir, flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = int(main() or 0)
    finally:
        _cleanup_runtime()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
    os._exit(exit_code)
