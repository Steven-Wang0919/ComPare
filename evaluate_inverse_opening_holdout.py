# -*- coding: utf-8 -*-
"""
evaluate_inverse_opening_holdout.py

Inverse leave-one-opening-out cross-validation with fair tuning audits.
"""

import os
import sys
import gc

import numpy as np
import pandas as pd

from common_utils import build_protocol_splits, load_data
from inverse_mlp import train_and_eval_inverse_mlp
from inverse_grnn import train_and_eval_inverse_grnn
from inverse_kan import train_and_eval_inverse_kan_v2
from policy_config import (
    POLICY_LABEL,
    POLICY_LOW_MID_THRESHOLD,
    POLICY_MID_HIGH_THRESHOLD,
    POLICY_TARGET_OPENINGS,
)
from run_utils import (
    append_manifest_outputs,
    create_run_dir,
    save_dataframe,
    write_manifest,
)

VAL_RATIO = 0.2


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


def _build_opening_folds(X, y, seed):
    unique_openings = _format_openings(np.unique(np.asarray(X)[:, 0]))
    folds = []
    for fold_idx, opening in enumerate(unique_openings, start=1):
        split_info = build_protocol_splits(
            X,
            y,
            protocol="leave_one_opening_out",
            random_state=seed,
            val_size=VAL_RATIO,
            holdout_opening=opening,
        )
        train_val_openings = [op for op in unique_openings if not np.isclose(op, opening, atol=1e-8)]
        fold_info = {
            "fold_id": fold_idx,
            "test_opening_mm": float(opening),
            "train_val_openings_mm": train_val_openings,
            "train_val_openings_label": _format_openings_str(train_val_openings),
            "train_size": int(len(split_info["idx_train"])),
            "val_size": int(len(split_info["idx_val"])),
            "test_size": int(len(split_info["idx_test"])),
            "split_info": split_info,
        }
        folds.append(fold_info)
    return unique_openings, folds


def _collect_fold_outputs(fold_info, model_results):
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
    _validate_same_values(
        strat_open_all_mlp,
        strat_open_all_grnn,
        "inverse_MLP strategy_opening_all",
        "inverse_GRNN strategy_opening_all",
    )
    _validate_same_values(
        strat_open_all_mlp,
        strat_open_all_kan,
        "inverse_MLP strategy_opening_all",
        "inverse_KAN strategy_opening_all",
    )
    _validate_same_mask(policy_mask_mlp, policy_mask_grnn, "inverse_MLP policy_mask", "inverse_GRNN policy_mask")
    _validate_same_mask(policy_mask_mlp, policy_mask_kan, "inverse_MLP policy_mask", "inverse_KAN policy_mask")

    test_indices = np.asarray(fold_info["split_info"]["idx_test"], dtype=int).reshape(-1)
    if len(test_indices) != len(y_true_all_mlp):
        raise ValueError(
            f"Fold {fold_info['fold_id']} has mismatched idx_test and predictions lengths: "
            f"{len(test_indices)} vs {len(y_true_all_mlp)}"
        )

    common_cols = {
        "fold_id": int(fold_info["fold_id"]),
        "test_opening_mm": float(fold_info["test_opening_mm"]),
        "train_val_openings_mm": fold_info["train_val_openings_label"],
        "sample_index": test_indices,
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
        "sample_index": test_indices[main_idx],
        "target_mass_g_min": mass_all_mlp[main_idx],
        "actual_opening_mm": opening_all_mlp[main_idx],
        "strategy_opening_mm": strat_open_all_mlp[main_idx],
        "true_speed_r_min": y_true_all_mlp[main_idx],
        "inverse_MLP_pred": y_pred_all_mlp[main_idx],
        "inverse_GRNN_pred": y_pred_all_grnn[main_idx],
        "inverse_KAN_pred": y_pred_all_kan[main_idx],
    })

    return df_all, df_main


def _save_tuning_audit(output_dir, tuning_rows):
    if len(tuning_rows) == 0:
        return None
    path = os.path.join(output_dir, "inverse_opening_holdout_tuning_audit.csv")
    save_dataframe(pd.DataFrame(tuning_rows), path)
    return path


def _format_hyperparams(model_name, result):
    if model_name == "inverse_MLP":
        return (
            f"hidden={result.get('best_hidden')}, "
            f"alpha={_fmt_float(result.get('best_alpha'))}"
        )
    if model_name == "inverse_GRNN":
        return f"sigma={_fmt_float(result.get('best_sigma'))}"
    return (
        f"hidden={result.get('best_hidden_dim')}, "
        f"lr={_fmt_float(result.get('best_lr'))}, "
        f"wd={_fmt_float(result.get('best_weight_decay'))}"
    )


def _make_summary(df_metrics):
    return (
        df_metrics.groupby(["Task", "Model", "DisplayName"], as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            R2_main_mean=("R2_main", "mean"),
            R2_main_std=("R2_main", "std"),
            R2_main_min=("R2_main", "min"),
            R2_main_max=("R2_main", "max"),
            ARE_main_pct_mean=("ARE_main(%)", "mean"),
            ARE_main_pct_std=("ARE_main(%)", "std"),
            ARE_main_pct_min=("ARE_main(%)", "min"),
            ARE_main_pct_max=("ARE_main(%)", "max"),
            R2_all_mean=("R2_all", "mean"),
            R2_all_std=("R2_all", "std"),
            R2_all_min=("R2_all", "min"),
            R2_all_max=("R2_all", "max"),
            ARE_all_pct_mean=("ARE_all(%)", "mean"),
            ARE_all_pct_std=("ARE_all(%)", "std"),
            ARE_all_pct_min=("ARE_all(%)", "min"),
            ARE_all_pct_max=("ARE_all(%)", "max"),
            n_main_mean=("n_main", "mean"),
            n_main_min=("n_main", "min"),
            n_main_max=("n_main", "max"),
            main_ratio_mean=("main_ratio", "mean"),
            main_ratio_std=("main_ratio", "std"),
            main_ratio_min=("main_ratio", "min"),
            main_ratio_max=("main_ratio", "max"),
            n_all_mean=("n_all", "mean"),
            n_all_min=("n_all", "min"),
            n_all_max=("n_all", "max"),
        )
        .sort_values("Model")
        .reset_index(drop=True)
    )


def _validate_cv_outputs(df_all, total_samples, expected_folds):
    observed_folds = int(df_all["fold_id"].nunique())
    if observed_folds != int(expected_folds):
        raise ValueError(f"Expected {expected_folds} folds, got {observed_folds}")
    if len(df_all) != int(total_samples):
        raise ValueError(f"Expected {total_samples} prediction rows, got {len(df_all)}")
    sample_indices = df_all["sample_index"].to_numpy(dtype=int)
    unique_indices = np.unique(sample_indices)
    if len(unique_indices) != int(total_samples):
        raise ValueError("Each sample must appear exactly once in predictions_all.")
    if sample_indices.min() != 0 or sample_indices.max() != int(total_samples) - 1:
        raise ValueError("Predictions_all sample_index range does not cover the full dataset.")


def run_inverse_opening_holdout_compare(output_dir, data_path="data/dataset.xlsx", seed=42):
    print("\n" + "=" * 72, flush=True)
    print("Inverse evaluation: leave-one-opening-out 7-fold CV", flush=True)
    print("=" * 72, flush=True)

    X_raw, y_raw = load_data(data_path)
    unique_openings, folds = _build_opening_folds(X_raw, y_raw, seed)

    metrics_rows = []
    predictions_all = []
    predictions_main = []
    tuning_rows = []
    artifact_outputs = []
    pred_all_relpath = "inverse_opening_holdout_predictions_all.csv"

    total_folds = len(folds)
    for fold_info in folds:
        fold_id = fold_info["fold_id"]
        test_opening = fold_info["test_opening_mm"]
        split_info = fold_info["split_info"]
        split_indices = (
            split_info["idx_train"],
            split_info["idx_val"],
            split_info["idx_test"],
        )

        print("\n" + "-" * 72, flush=True)
        print(
            f"Fold {fold_id}/{total_folds} | "
            f"test opening = {test_opening:g} mm | "
            f"train/val openings = {fold_info['train_val_openings_label']}",
            flush=True,
        )
        print(
            f"train={fold_info['train_size']}, "
            f"val={fold_info['val_size']}, "
            f"test={fold_info['test_size']}",
            flush=True,
        )
        print("-" * 72, flush=True)

        mlp_res = train_and_eval_inverse_mlp(
            data_path=data_path,
            random_state=seed,
            save_outputs_dir=None,
            split_indices=split_indices,
            save_artifacts=True,
            artifact_dir=os.path.join(
                output_dir,
                "artifacts",
                f"fold_{int(fold_id):02d}_opening_{_fmt_float(test_opening)}",
                "inverse_MLP",
            ),
            save_test_slice=True,
            artifact_extra={
                "run_dir": output_dir.replace("\\", "/"),
                "fold_id": int(fold_id),
                "test_opening_mm": float(test_opening),
                "train_val_openings_mm": fold_info["train_val_openings_label"],
                "reference_output": {
                    "path": pred_all_relpath,
                    "filter_column": "fold_id",
                    "filter_value": int(fold_id),
                    "prediction_column": "inverse_MLP_pred",
                    "target_column": "true_speed_r_min",
                },
            },
        )
        grnn_res = train_and_eval_inverse_grnn(
            data_path=data_path,
            save_outputs_dir=None,
            split_indices=split_indices,
            random_state=seed,
            save_artifacts=True,
            artifact_dir=os.path.join(
                output_dir,
                "artifacts",
                f"fold_{int(fold_id):02d}_opening_{_fmt_float(test_opening)}",
                "inverse_GRNN",
            ),
            save_test_slice=True,
            artifact_extra={
                "run_dir": output_dir.replace("\\", "/"),
                "fold_id": int(fold_id),
                "test_opening_mm": float(test_opening),
                "train_val_openings_mm": fold_info["train_val_openings_label"],
                "reference_output": {
                    "path": pred_all_relpath,
                    "filter_column": "fold_id",
                    "filter_value": int(fold_id),
                    "prediction_column": "inverse_GRNN_pred",
                    "target_column": "true_speed_r_min",
                },
            },
        )
        kan_res = train_and_eval_inverse_kan_v2(
            data_path=data_path,
            seed=seed,
            save_artifacts=True,
            artifact_dir=os.path.join(
                output_dir,
                "artifacts",
                f"fold_{int(fold_id):02d}_opening_{_fmt_float(test_opening)}",
                "inverse_KAN",
            ),
            save_outputs_dir=None,
            split_indices=split_indices,
            save_test_slice=True,
            artifact_extra={
                "run_dir": output_dir.replace("\\", "/"),
                "fold_id": int(fold_id),
                "test_opening_mm": float(test_opening),
                "train_val_openings_mm": fold_info["train_val_openings_label"],
                "reference_output": {
                    "path": pred_all_relpath,
                    "filter_column": "fold_id",
                    "filter_value": int(fold_id),
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
            metrics_rows.append(
                {
                    "Task": "inverse_opening_holdout",
                    "fold_id": int(fold_id),
                    "test_opening_mm": float(test_opening),
                    "train_val_openings_mm": fold_info["train_val_openings_label"],
                    "train_size": int(fold_info["train_size"]),
                    "val_size": int(fold_info["val_size"]),
                    "test_size": int(fold_info["test_size"]),
                    "Model": model_name,
                    "DisplayName": model_name,
                    "ArchitectureNote": f"leave-one-opening-out; test opening = {test_opening:g} mm",
                    "R2_main": result["r2_main"],
                    "ARE_main(%)": result["are_main"],
                    "n_main": result["n_main"],
                    "main_ratio": result["main_ratio"],
                    "R2_all": result["r2_all"],
                    "ARE_all(%)": result["are_all"],
                    "n_all": result["n_all"],
                    "Hyperparams": _format_hyperparams(model_name, result),
                }
            )

            for row in result.get("tuning_records", []):
                merged = {
                    "evaluation_scope": "inverse_opening_holdout",
                    "fold_id": int(fold_id),
                    "test_opening_mm": float(test_opening),
                    "train_val_openings_mm": fold_info["train_val_openings_label"],
                    "reported_model": model_name,
                }
                merged.update(row)
                tuning_rows.append(merged)

        df_fold_all, df_fold_main = _collect_fold_outputs(fold_info, fold_results)
        predictions_all.append(df_fold_all)
        predictions_main.append(df_fold_main)
        for result in fold_results.values():
            artifact_outputs.extend(_artifact_outputs_from_result(result, output_dir))
        print(f"Fold {fold_id}/{total_folds} done.", flush=True)

    df_metrics = pd.DataFrame(metrics_rows)
    df_summary = _make_summary(df_metrics)
    df_all = pd.concat(predictions_all, ignore_index=True).sort_values(
        ["sample_index", "fold_id"]
    ).reset_index(drop=True)
    df_main = pd.concat(predictions_main, ignore_index=True)
    _validate_cv_outputs(df_all, total_samples=len(X_raw), expected_folds=total_folds)

    metrics_path = os.path.join(output_dir, "inverse_opening_holdout_metrics.csv")
    summary_path = os.path.join(output_dir, "inverse_opening_holdout_summary.csv")
    all_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_all.csv")
    main_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_main.csv")

    save_dataframe(df_metrics, metrics_path)
    save_dataframe(df_summary, summary_path)
    save_dataframe(df_all, all_path)
    save_dataframe(df_main, main_path)

    tuning_audit_path = _save_tuning_audit(output_dir, tuning_rows)

    print(f"Completed {len(unique_openings)} opening holdout folds.", flush=True)
    print(f"Saved fold metrics: {metrics_path}", flush=True)
    print(f"Saved summary metrics: {summary_path}", flush=True)
    print(f"Saved all predictions: {all_path}", flush=True)
    print(f"Saved main-scope predictions: {main_path}", flush=True)

    return {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "all_path": all_path,
        "main_path": main_path,
        "tuning_audit_path": tuning_audit_path,
        "artifact_outputs": artifact_outputs,
    }


def main():
    data_path = "data/dataset.xlsx"
    seed = 42
    X_raw, _ = load_data(data_path)
    unique_openings = _format_openings(np.unique(X_raw[:, 0]))
    run_dir = create_run_dir("evaluate_inverse_opening_holdout")

    manifest_path = write_manifest(
        run_dir,
        script_name="evaluate_inverse_opening_holdout.py",
        data_path=data_path,
        seed=seed,
        params={
            "protocol": "leave_one_opening_out_cv",
            "fold_openings_mm": unique_openings,
            "n_folds": len(unique_openings),
            "val_ratio": VAL_RATIO,
            "policy": {
                "label": POLICY_LABEL,
                "target_openings_mm": list(POLICY_TARGET_OPENINGS),
                "threshold_low_mid": POLICY_LOW_MID_THRESHOLD,
                "threshold_mid_high": POLICY_MID_HIGH_THRESHOLD,
            },
            "note": "each fold holds out exactly one opening and evaluates all seven openings once",
            "fair_tuning": {
                "n_candidates": 24,
                "n_repeats": 5,
                "selection_metric": "mean_val_r2",
                "tie_break_metric": "mean_val_are",
                "budget_profile": "high",
            },
        },
        source_files=_artifact_source_files(),
    )

    print(f"\nRun directory: {run_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    outputs = run_inverse_opening_holdout_compare(run_dir, data_path=data_path, seed=seed)

    manifest_outputs = [
        {"path": os.path.relpath(outputs["metrics_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["summary_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["all_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["main_path"], run_dir).replace("\\", "/")},
    ]
    if outputs["tuning_audit_path"] is not None:
        manifest_outputs.append(
            {"path": os.path.relpath(outputs["tuning_audit_path"], run_dir).replace("\\", "/")}
        )
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
