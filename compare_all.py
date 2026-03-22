# -*- coding: utf-8 -*-
"""
compare_all.py

Unified forward/inverse comparison with fair tuning audit outputs.
"""

import os

import numpy as np
import pandas as pd

from train_mlp import train_and_eval_mlp
from train_grnn import train_and_eval_grnn
from train_kan import train_and_eval_kan

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


def _to_1d_array(arr, name):
    out = np.asarray(arr).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} is empty.")
    return out


def _validate_same_length(arr1, arr2, name1, name2):
    if len(arr1) != len(arr2):
        raise ValueError(f"{name1} and {name2} have different lengths.")


def _validate_same_values(arr1, arr2, name1, name2, atol=1e-8, rtol=1e-6):
    if not np.allclose(arr1, arr2, atol=atol, rtol=rtol, equal_nan=False):
        bad = np.where(~np.isclose(arr1, arr2, atol=atol, rtol=rtol, equal_nan=False))[0]
        idx = int(bad[0])
        raise ValueError(
            f"{name1} and {name2} differ at index {idx}: "
            f"{name1}={arr1[idx]}, {name2}={arr2[idx]}"
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


def _save_tuning_audit(output_dir, filename, model_results, extra_cols=None):
    rows = []
    extra_cols = extra_cols or {}
    for model_name, result in model_results.items():
        for row in result.get("tuning_records", []):
            merged = dict(extra_cols)
            merged["reported_model"] = model_name
            merged.update(row)
            rows.append(merged)

    if len(rows) == 0:
        return None

    audit_path = os.path.join(output_dir, filename)
    save_dataframe(pd.DataFrame(rows), audit_path)
    return audit_path


def run_forward_compare(output_dir, data_path="data/dataset.xlsx", seed=42):
    print("\n" + "=" * 72)
    print("开始正向模型对比：MLP / GRNN / KAN")
    print("=" * 72)

    mlp_res = train_and_eval_mlp(
        data_path=data_path,
        random_state=seed,
        save_csv_path=None,
    )
    grnn_res = train_and_eval_grnn(
        data_path=data_path,
        save_csv_path=None,
        random_state=seed,
    )
    kan_res = train_and_eval_kan(
        data_path=data_path,
        seed=seed,
        save_csv_path=None,
        save_artifacts=False,
    )

    metrics = [
        {
            "Task": "forward",
            "Model": "MLP",
            "DisplayName": "MLP",
            "ArchitectureNote": "",
            "R2": mlp_res["r2"],
            "ARE(%)": mlp_res["are"],
            "Hyperparams": (
                f"hidden={mlp_res.get('best_hidden')}, "
                f"alpha={mlp_res.get('best_alpha')}"
            ),
        },
        {
            "Task": "forward",
            "Model": "GRNN",
            "DisplayName": "GRNN",
            "ArchitectureNote": "",
            "R2": grnn_res["r2"],
            "ARE(%)": grnn_res["are"],
            "Hyperparams": f"sigma={_fmt_float(grnn_res.get('best_sigma'))}",
        },
        {
            "Task": "forward",
            "Model": "KAN",
            "DisplayName": "KAN",
            "ArchitectureNote": "forward spline/grid KAN",
            "R2": kan_res["r2"],
            "ARE(%)": kan_res["are"],
            "Hyperparams": (
                f"hidden={kan_res.get('best_hidden_dim')}, "
                f"lr={_fmt_float(kan_res.get('best_lr'))}, "
                f"wd={_fmt_float(kan_res.get('best_weight_decay'))}"
            ),
        },
    ]

    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "forward_model_metrics.csv")
    save_dataframe(df_metrics, metrics_path)

    y_true_mlp = _to_1d_array(mlp_res["y_true"], "MLP y_true")
    y_true_grnn = _to_1d_array(grnn_res["y_true"], "GRNN y_true")
    y_true_kan = _to_1d_array(kan_res["y_true"], "KAN y_true")

    y_pred_mlp = _to_1d_array(mlp_res["y_pred"], "MLP y_pred")
    y_pred_grnn = _to_1d_array(grnn_res["y_pred"], "GRNN y_pred")
    y_pred_kan = _to_1d_array(kan_res["y_pred"], "KAN y_pred")

    _validate_same_length(y_true_mlp, y_pred_mlp, "MLP y_true", "MLP y_pred")
    _validate_same_length(y_true_grnn, y_pred_grnn, "GRNN y_true", "GRNN y_pred")
    _validate_same_length(y_true_kan, y_pred_kan, "KAN y_true", "KAN y_pred")

    _validate_same_length(y_true_mlp, y_true_grnn, "MLP y_true", "GRNN y_true")
    _validate_same_length(y_true_mlp, y_true_kan, "MLP y_true", "KAN y_true")

    _validate_same_values(y_true_mlp, y_true_grnn, "MLP y_true", "GRNN y_true")
    _validate_same_values(y_true_mlp, y_true_kan, "MLP y_true", "KAN y_true")

    df_pred = pd.DataFrame({
        "true": y_true_mlp,
        "MLP_pred": y_pred_mlp,
        "GRNN_pred": y_pred_grnn,
        "KAN_pred": y_pred_kan,
    })
    pred_path = os.path.join(output_dir, "forward_model_predictions.csv")
    save_dataframe(df_pred, pred_path)

    tuning_audit_path = _save_tuning_audit(
        output_dir,
        "forward_tuning_audit.csv",
        {
            "MLP": mlp_res,
            "GRNN": grnn_res,
            "KAN": kan_res,
        },
        extra_cols={"comparison_scope": "forward"},
    )

    print(f"正向指标已保存：{metrics_path}")
    print(f"正向预测已保存：{pred_path}")

    return {
        "metrics_path": metrics_path,
        "pred_path": pred_path,
        "tuning_audit_path": tuning_audit_path,
    }


def run_inverse_compare(output_dir, data_path="data/dataset.xlsx", seed=42):
    print("\n" + "=" * 72)
    print("开始反向模型对比：inverse_MLP / inverse_GRNN / inverse_KAN")
    print("=" * 72)

    mlp_res = train_and_eval_inverse_mlp(
        data_path=data_path,
        random_state=seed,
        save_outputs_dir=None,
    )
    grnn_res = train_and_eval_inverse_grnn(
        data_path=data_path,
        save_outputs_dir=None,
        random_state=seed,
    )
    kan_res = train_and_eval_inverse_kan_v2(
        data_path=data_path,
        seed=seed,
        save_artifacts=False,
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
            "Task": "inverse",
            "Model": "inverse_MLP",
            "DisplayName": "inverse_MLP",
            "ArchitectureNote": "",
            "R2_main": mlp_res["r2_main"],
            "ARE_main(%)": mlp_res["are_main"],
            "n_main": mlp_res["n_main"],
            "main_ratio": mlp_res["main_ratio"],
            "R2_all": mlp_res["r2_all"],
            "ARE_all(%)": mlp_res["are_all"],
            "n_all": mlp_res["n_all"],
            "Hyperparams": (
                f"hidden={mlp_res.get('best_hidden')}, "
                f"alpha={mlp_res.get('best_alpha')}"
            ),
        },
        {
            "Task": "inverse",
            "Model": "inverse_GRNN",
            "DisplayName": "inverse_GRNN",
            "ArchitectureNote": "",
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
            "Task": "inverse",
            "Model": "inverse_KAN",
            "DisplayName": "inverse_KAN",
            "ArchitectureNote": "repaired spline/grid KAN",
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
    metrics_path = os.path.join(output_dir, "inverse_model_metrics.csv")
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
    all_path = os.path.join(output_dir, "inverse_model_predictions_all.csv")
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
    main_path = os.path.join(output_dir, "inverse_model_predictions_main.csv")
    save_dataframe(df_main, main_path)

    tuning_audit_path = _save_tuning_audit(
        output_dir,
        "inverse_tuning_audit.csv",
        {
            "inverse_MLP": mlp_res,
            "inverse_GRNN": grnn_res,
            "inverse_KAN": kan_res,
        },
        extra_cols={"comparison_scope": "inverse"},
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
    run_dir = create_run_dir("compare_all")

    manifest_path = write_manifest(
        run_dir,
        script_name="compare_all.py",
        data_path=data_path,
        seed=seed,
        params={
            "forward_models": ["MLP", "GRNN", "KAN"],
            "inverse_models": ["inverse_MLP", "inverse_GRNN", "inverse_KAN"],
            "save_root": "runs",
            "policy": {
                "label": POLICY_LABEL,
                "target_openings_mm": list(POLICY_TARGET_OPENINGS),
                "threshold_low_mid": POLICY_LOW_MID_THRESHOLD,
                "threshold_mid_high": POLICY_MID_HIGH_THRESHOLD,
            },
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

    forward_outputs = run_forward_compare(run_dir, data_path=data_path, seed=seed)
    inverse_outputs = run_inverse_compare(run_dir, data_path=data_path, seed=seed)

    append_manifest_outputs(
        run_dir,
        [
            {"path": os.path.relpath(forward_outputs["metrics_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(forward_outputs["pred_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(forward_outputs["tuning_audit_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(inverse_outputs["metrics_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(inverse_outputs["all_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(inverse_outputs["main_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(inverse_outputs["tuning_audit_path"], run_dir).replace("\\", "/")},
        ],
    )

    print("\n全部结果已统一输出到：")
    print(run_dir)


if __name__ == "__main__":
    main()
