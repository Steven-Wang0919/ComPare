# -*- coding: utf-8 -*-
"""
evaluate_inverse_opening_holdout.py

Inverse opening-holdout evaluation with fair tuning audits.
"""

import os

import numpy as np
import pandas as pd

from common_utils import build_opening_holdout_indices, load_data
from inverse_mlp import train_and_eval_inverse_mlp
from inverse_grnn import train_and_eval_inverse_grnn
from inverse_kan import train_and_eval_inverse_kan_v2
from run_utils import (
    append_manifest_outputs,
    create_run_dir,
    save_dataframe,
    write_manifest,
)


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0
TARGET_OPENINGS = (20.0, 35.0, 50.0)
OPENING_ATOL = 0.1


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


def _save_tuning_audit(output_dir, model_results):
    rows = []
    for model_name, result in model_results.items():
        for row in result.get("tuning_records", []):
            merged = {
                "evaluation_scope": "inverse_opening_holdout",
                "reported_model": model_name,
            }
            merged.update(row)
            rows.append(merged)

    if len(rows) == 0:
        return None

    path = os.path.join(output_dir, "inverse_opening_holdout_tuning_audit.csv")
    save_dataframe(pd.DataFrame(rows), path)
    return path


def run_inverse_opening_holdout_compare(output_dir, data_path="data/dataset.xlsx", seed=42):
    print("\n" + "=" * 72)
    print("开始反向模型对比：opening-holdout (test = all 20/35/50 mm)")
    print("=" * 72)

    X_raw, y_raw = load_data(data_path)
    split_indices = build_opening_holdout_indices(
        X_raw,
        y_raw,
        random_state=seed,
        val_ratio=0.2,
        target_openings=TARGET_OPENINGS,
        atol=OPENING_ATOL,
    )

    mlp_res = train_and_eval_inverse_mlp(
        data_path=data_path,
        random_state=seed,
        save_outputs_dir=None,
        split_indices=split_indices,
    )
    grnn_res = train_and_eval_inverse_grnn(
        data_path=data_path,
        save_outputs_dir=None,
        split_indices=split_indices,
        random_state=seed,
    )
    kan_res = train_and_eval_inverse_kan_v2(
        data_path=data_path,
        seed=seed,
        save_artifacts=False,
        save_outputs_dir=None,
        split_indices=split_indices,
    )

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

    metrics = [
        {
            "Task": "inverse_opening_holdout",
            "Model": "inverse_MLP",
            "DisplayName": "inverse_MLP",
            "ArchitectureNote": "test = all 20/35/50 mm openings",
            "R2_main": mlp_res["r2_main"],
            "ARE_main(%)": mlp_res["are_main"],
            "n_main": mlp_res["n_main"],
            "main_ratio": mlp_res["main_ratio"],
            "R2_all": mlp_res["r2_all"],
            "ARE_all(%)": mlp_res["are_all"],
            "n_all": mlp_res["n_all"],
            "Hyperparams": (
                f"hidden={mlp_res.get('best_hidden')}, "
                f"alpha={_fmt_float(mlp_res.get('best_alpha'))}"
            ),
        },
        {
            "Task": "inverse_opening_holdout",
            "Model": "inverse_GRNN",
            "DisplayName": "inverse_GRNN",
            "ArchitectureNote": "test = all 20/35/50 mm openings",
            "R2_main": grnn_res["r2_main"],
            "ARE_main(%)": grnn_res["are_main"],
            "n_main": grnn_res["n_main"],
            "main_ratio": grnn_res["main_ratio"],
            "R2_all": grnn_res["r2_all"],
            "ARE_all(%)": grnn_res["are_all"],
            "n_all": grnn_res["n_all"],
            "Hyperparams": f"sigma={_fmt_float(grnn_res.get('best_sigma'))}",
        },
        {
            "Task": "inverse_opening_holdout",
            "Model": "inverse_KAN",
            "DisplayName": "inverse_KAN",
            "ArchitectureNote": "repaired spline/grid KAN; test = all 20/35/50 mm openings",
            "R2_main": kan_res["r2_main"],
            "ARE_main(%)": kan_res["are_main"],
            "n_main": kan_res["n_main"],
            "main_ratio": kan_res["main_ratio"],
            "R2_all": kan_res["r2_all"],
            "ARE_all(%)": kan_res["are_all"],
            "n_all": kan_res["n_all"],
            "Hyperparams": (
                f"hidden={kan_res.get('best_hidden_dim')}, "
                f"lr={_fmt_float(kan_res.get('best_lr'))}, "
                f"wd={_fmt_float(kan_res.get('best_weight_decay'))}"
            ),
        },
    ]
    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "inverse_opening_holdout_metrics.csv")
    save_dataframe(df_metrics, metrics_path)

    df_all = pd.DataFrame({
        "target_mass_g_min": mass_all_mlp,
        "actual_opening_mm": opening_all_mlp,
        "strategy_opening_mm": strat_open_all_mlp,
        "true_speed_r_min": y_true_all_mlp,
        "inverse_MLP_pred": y_pred_all_mlp,
        "inverse_GRNN_pred": y_pred_all_grnn,
        "inverse_KAN_pred": y_pred_all_kan,
        "policy_match": policy_mask_mlp.astype(int),
    })
    all_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_all.csv")
    save_dataframe(df_all, all_path)

    main_idx = np.where(policy_mask_mlp)[0]
    df_main = pd.DataFrame({
        "target_mass_g_min": mass_all_mlp[main_idx],
        "actual_opening_mm": opening_all_mlp[main_idx],
        "strategy_opening_mm": strat_open_all_mlp[main_idx],
        "true_speed_r_min": y_true_all_mlp[main_idx],
        "inverse_MLP_pred": y_pred_all_mlp[main_idx],
        "inverse_GRNN_pred": y_pred_all_grnn[main_idx],
        "inverse_KAN_pred": y_pred_all_kan[main_idx],
    })
    main_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_main.csv")
    save_dataframe(df_main, main_path)

    tuning_audit_path = _save_tuning_audit(
        output_dir,
        {
            "inverse_MLP": mlp_res,
            "inverse_GRNN": grnn_res,
            "inverse_KAN": kan_res,
        },
    )

    print(f"反向指标已保存：{metrics_path}")
    print(f"反向全测试集预测已保存：{all_path}")
    print(f"反向主结果子集预测已保存：{main_path}")

    return {
        "metrics_path": metrics_path,
        "all_path": all_path,
        "main_path": main_path,
        "tuning_audit_path": tuning_audit_path,
    }


def main():
    data_path = "data/dataset.xlsx"
    seed = 42
    run_dir = create_run_dir("evaluate_inverse_opening_holdout")

    manifest_path = write_manifest(
        run_dir,
        script_name="evaluate_inverse_opening_holdout.py",
        data_path=data_path,
        seed=seed,
        params={
            "test_openings_mm": list(TARGET_OPENINGS),
            "threshold_low_mid": THRESHOLD_LOW_MID,
            "threshold_mid_high": THRESHOLD_MID_HIGH,
            "note": "all 20/35/50 mm samples are excluded from training and used only as inverse-task test set",
            "fair_tuning": {
                "n_candidates": 24,
                "n_repeats": 5,
                "selection_metric": "mean_val_r2",
                "tie_break_metric": "mean_val_are",
                "budget_profile": "high",
            },
        },
    )

    print(f"\n本次运行目录：{run_dir}")
    print(f"Manifest：{manifest_path}")

    outputs = run_inverse_opening_holdout_compare(run_dir, data_path=data_path, seed=seed)

    manifest_outputs = [
        {"path": os.path.relpath(outputs["metrics_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["all_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(outputs["main_path"], run_dir).replace("\\", "/")},
    ]
    if outputs["tuning_audit_path"] is not None:
        manifest_outputs.append(
            {"path": os.path.relpath(outputs["tuning_audit_path"], run_dir).replace("\\", "/")}
        )

    append_manifest_outputs(run_dir, manifest_outputs)

    print("\n全部结果已统一输出到：")
    print(run_dir)


if __name__ == "__main__":
    main()
