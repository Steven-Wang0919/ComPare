# -*- coding: utf-8 -*-
"""
evaluate_generalization.py

Generalization evaluation across multiple protocols with fair tuning audits.
"""

import os

import numpy as np
import pandas as pd

from common_utils import build_protocol_splits, load_data
from train_mlp import train_and_eval_mlp
from train_grnn import train_and_eval_grnn
from train_kan import train_and_eval_kan
from run_utils import append_manifest_outputs, create_run_dir, save_dataframe, write_manifest


MODEL_RUNNERS = {
    "MLP": lambda data_path, seed, split_indices: train_and_eval_mlp(
        data_path=data_path,
        random_state=seed,
        split_indices=split_indices,
        save_csv_path=None,
    ),
    "GRNN": lambda data_path, seed, split_indices: train_and_eval_grnn(
        data_path=data_path,
        split_indices=split_indices,
        save_csv_path=None,
        random_state=seed,
    ),
    "KAN": lambda data_path, seed, split_indices: train_and_eval_kan(
        data_path=data_path,
        seed=seed,
        split_indices=split_indices,
        save_csv_path=None,
        save_artifacts=False,
    ),
}


def _compact_protocol_meta(split_info):
    out = {}
    for k, v in split_info.items():
        if k.startswith("idx_"):
            continue
        if isinstance(v, (np.integer, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _run_one_protocol(name, split_info, data_path, seed):
    split_indices = (
        split_info["idx_train"],
        split_info["idx_val"],
        split_info["idx_test"],
    )

    results = {}
    for model_name, runner in MODEL_RUNNERS.items():
        print(f"\n--- [{name}] 运行模型：{model_name} ---")
        results[model_name] = runner(data_path, seed, split_indices)

    ref_model = next(iter(results.keys()))
    ref_res = results[ref_model]

    pred_df = pd.DataFrame({
        "protocol": name,
        "opening": ref_res["x_test_raw"][:, 0],
        "speed": ref_res["x_test_raw"][:, 1],
        "true": ref_res["y_true"],
    })
    for model_name, res in results.items():
        pred_df[f"{model_name}_pred"] = res["y_pred"]

    metrics_rows = []
    tuning_rows = []
    for model_name, res in results.items():
        row = {
            "protocol": name,
            "protocol_desc": split_info.get("description", ""),
            "model": model_name,
            "r2": res["r2"],
            "are": res["are"],
            "train_size": len(split_info["idx_train"]),
            "val_size": len(split_info["idx_val"]),
            "test_size": len(split_info["idx_test"]),
        }
        for k, v in _compact_protocol_meta(split_info).items():
            if k in {"protocol", "description"}:
                continue
            row[k] = v
        metrics_rows.append(row)

        for tuning_row in res.get("tuning_records", []):
            merged = {
                "protocol": name,
                "protocol_desc": split_info.get("description", ""),
                "model": model_name,
            }
            merged.update(tuning_row)
            tuning_rows.append(merged)

    return pd.DataFrame(metrics_rows), pred_df, pd.DataFrame(tuning_rows)


def _build_all_protocols(X, y, seed):
    protocols = {}

    protocols["random_interp"] = build_protocol_splits(
        X,
        y,
        protocol="random_stratified",
        random_state=seed,
    )

    unique_openings = sorted(np.unique(X[:, 0]).tolist())
    for op in unique_openings:
        op_name = int(op) if float(op).is_integer() else op
        protocols[f"leave_opening_{op_name}_out"] = build_protocol_splits(
            X,
            y,
            protocol="leave_one_opening_out",
            random_state=seed,
            holdout_opening=op,
        )

    speed_blocks = [
        ("leave_speed_20_24_out", 20, 24),
        ("leave_speed_38_42_out", 38, 42),
        ("leave_speed_56_60_out", 56, 60),
    ]
    for protocol_name, smin, smax in speed_blocks:
        protocols[protocol_name] = build_protocol_splits(
            X,
            y,
            protocol="leave_speed_block_out",
            random_state=seed,
            holdout_speed_min=smin,
            holdout_speed_max=smax,
        )

    return protocols


def _make_protocol_summary(df_metrics):
    df_summary = df_metrics.pivot_table(
        index="protocol",
        columns="model",
        values=["r2", "are"],
    )
    df_summary.columns = [f"{a}_{b}" for a, b in df_summary.columns]
    return df_summary.reset_index()


def _make_protocol_family_summary(df_metrics):
    df = df_metrics.copy()

    def family_name(p):
        if p == "random_interp":
            return "random_interp"
        if str(p).startswith("leave_opening_"):
            return "leave_one_opening_out"
        if str(p).startswith("leave_speed_"):
            return "leave_speed_block_out"
        return "other"

    df["protocol_family"] = df["protocol"].map(family_name)
    return (
        df.groupby(["protocol_family", "model"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_min=("r2", "min"),
            r2_max=("r2", "max"),
            are_mean=("are", "mean"),
            are_std=("are", "std"),
            are_min=("are", "min"),
            are_max=("are", "max"),
            n_protocols=("protocol", "count"),
        )
        .sort_values(["protocol_family", "model"])
        .reset_index(drop=True)
    )


def _make_opening_cv_summary(df_metrics):
    df = df_metrics[df_metrics["protocol"].str.startswith("leave_opening_")].copy()
    if len(df) == 0:
        return pd.DataFrame()
    return (
        df.groupby("model", as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_min=("r2", "min"),
            r2_max=("r2", "max"),
            are_mean=("are", "mean"),
            are_std=("are", "std"),
            are_min=("are", "min"),
            are_max=("are", "max"),
            folds=("protocol", "count"),
        )
        .sort_values("r2_mean", ascending=False)
        .reset_index(drop=True)
    )


def _make_speed_cv_summary(df_metrics):
    df = df_metrics[df_metrics["protocol"].str.startswith("leave_speed_")].copy()
    if len(df) == 0:
        return pd.DataFrame()
    return (
        df.groupby("model", as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_min=("r2", "min"),
            r2_max=("r2", "max"),
            are_mean=("are", "mean"),
            are_std=("are", "std"),
            are_min=("are", "min"),
            are_max=("are", "max"),
            folds=("protocol", "count"),
        )
        .sort_values("r2_mean", ascending=False)
        .reset_index(drop=True)
    )


def main():
    data_path = "data/dataset.xlsx"
    seed = 42
    X, y = load_data(data_path)
    protocols = _build_all_protocols(X, y, seed)

    run_dir = create_run_dir("evaluate_generalization")
    write_manifest(
        run_dir,
        script_name="evaluate_generalization.py",
        data_path=data_path,
        seed=seed,
        params={
            "models": list(MODEL_RUNNERS.keys()),
            "protocols": {k: _compact_protocol_meta(v) for k, v in protocols.items()},
            "fair_tuning": {
                "n_candidates": 24,
                "n_repeats": 5,
                "selection_metric": "mean_val_r2",
                "tie_break_metric": "mean_val_are",
                "budget_profile": "high",
            },
        },
    )

    metrics_all = []
    preds_all = []
    tuning_all = []

    for name, split_info in protocols.items():
        print("\n" + "=" * 80)
        print(f"运行协议：{name} | {split_info.get('description', '')}")
        print("=" * 80)
        mdf, pdf, tdf = _run_one_protocol(name, split_info, data_path, seed)
        metrics_all.append(mdf)
        preds_all.append(pdf)
        tuning_all.append(tdf)

    df_metrics = pd.concat(metrics_all, ignore_index=True)
    df_preds = pd.concat(preds_all, ignore_index=True)
    df_tuning = pd.concat(tuning_all, ignore_index=True) if len(tuning_all) > 0 else pd.DataFrame()
    df_protocol_summary = _make_protocol_summary(df_metrics)
    df_family_summary = _make_protocol_family_summary(df_metrics)
    df_opening_cv_summary = _make_opening_cv_summary(df_metrics)
    df_speed_cv_summary = _make_speed_cv_summary(df_metrics)

    save_dataframe(df_metrics, os.path.join(run_dir, "protocol_metrics.csv"))
    save_dataframe(df_preds, os.path.join(run_dir, "protocol_predictions.csv"))
    save_dataframe(df_protocol_summary, os.path.join(run_dir, "protocol_summary.csv"))
    save_dataframe(df_family_summary, os.path.join(run_dir, "protocol_family_summary.csv"))
    save_dataframe(df_opening_cv_summary, os.path.join(run_dir, "opening_cv_summary.csv"))
    save_dataframe(df_speed_cv_summary, os.path.join(run_dir, "speed_cv_summary.csv"))
    if len(df_tuning) > 0:
        save_dataframe(df_tuning, os.path.join(run_dir, "protocol_tuning_audit.csv"))

    outputs = [
        {"path": "protocol_metrics.csv"},
        {"path": "protocol_predictions.csv"},
        {"path": "protocol_summary.csv"},
        {"path": "protocol_family_summary.csv"},
        {"path": "opening_cv_summary.csv"},
        {"path": "speed_cv_summary.csv"},
    ]
    if len(df_tuning) > 0:
        outputs.append({"path": "protocol_tuning_audit.csv"})

    append_manifest_outputs(run_dir, outputs)
    print(f"\n结果目录：{run_dir}")


if __name__ == "__main__":
    main()
