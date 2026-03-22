# -*- coding: utf-8 -*-
"""
evaluate_generalization.py

Generalization evaluation across multiple protocols with fair tuning audits.
"""

import os

import numpy as np
import pandas as pd

from common_utils import build_protocol_splits, load_data
from fair_tuning import build_protocol_aligned_inner_splits
from train_mlp import train_and_eval_mlp
from train_grnn import train_and_eval_grnn
from train_kan import train_and_eval_kan
from run_utils import append_manifest_outputs, create_run_dir, save_dataframe, write_manifest


SPEED_BLOCK_SIZE = 5
MODEL_NAMES = ("MLP", "GRNN", "KAN")


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


def _protocol_family(protocol_name):
    if protocol_name == "random_interp":
        return "random_interp"
    if str(protocol_name).startswith("leave_opening_"):
        return "leave_one_opening_out"
    if str(protocol_name).startswith("leave_speed_"):
        return "leave_speed_block_out"
    return "other"


def _build_inner_tuning_spec(X, split_info):
    protocol = str(split_info.get("protocol", ""))
    if protocol == "random_stratified":
        return "repeated_random", {}

    dev_idx = np.sort(np.concatenate([split_info["idx_train"], split_info["idx_val"]]))
    X_dev = np.asarray(X)[dev_idx]
    inner_splits, inner_split_strategy, inner_split_meta = build_protocol_aligned_inner_splits(
        X_dev,
        split_info=split_info,
        reference_X=X,
        speed_block_size=SPEED_BLOCK_SIZE,
    )
    return inner_split_strategy, {
        "inner_splits": inner_splits,
        "inner_split_strategy": inner_split_strategy,
        "inner_split_meta": inner_split_meta,
    }


def _run_model(model_name, data_path, seed, split_indices, inner_tuning_kwargs):
    common_kwargs = dict(
        data_path=data_path,
        split_indices=split_indices,
        save_csv_path=None,
        **inner_tuning_kwargs,
    )
    if model_name == "MLP":
        return train_and_eval_mlp(
            random_state=seed,
            **common_kwargs,
        )
    if model_name == "GRNN":
        return train_and_eval_grnn(
            random_state=seed,
            **common_kwargs,
        )
    if model_name == "KAN":
        return train_and_eval_kan(
            seed=seed,
            save_artifacts=False,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _run_one_protocol(name, split_info, data_path, seed, X):
    split_indices = (
        split_info["idx_train"],
        split_info["idx_val"],
        split_info["idx_test"],
    )
    inner_split_strategy, inner_tuning_kwargs = _build_inner_tuning_spec(X, split_info)

    results = {}
    for model_name in MODEL_NAMES:
        print(f"\n--- [{name}] Running model: {model_name} ---")
        results[model_name] = _run_model(
            model_name,
            data_path,
            seed,
            split_indices,
            inner_tuning_kwargs,
        )

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
            "inner_split_strategy": res.get("inner_split_strategy", inner_split_strategy),
            "inner_fold_count": res.get("inner_fold_count"),
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
    df["protocol_family"] = df["protocol"].map(_protocol_family)
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


def _concat_nonempty_frames(frames):
    valid_frames = [
        df for df in frames
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty
    ]
    if len(valid_frames) == 0:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


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
            "models": list(MODEL_NAMES),
            "protocols": {
                k: {
                    **_compact_protocol_meta(v),
                    "inner_split_strategy": _build_inner_tuning_spec(X, v)[0],
                }
                for k, v in protocols.items()
            },
            "fair_tuning": {
                "n_candidates": 24,
                "n_repeats": 5,
                "selection_metric": "mean_val_r2",
                "tie_break_metric": "mean_val_are",
                "budget_profile": "high",
                "random_interp_inner_protocol": "repeated_random",
                "holdout_inner_protocols": {
                    "leave_one_opening_out": "protocol_aligned_opening_cv",
                    "leave_speed_block_out": "protocol_aligned_speed_block_cv",
                },
                "speed_block_size": SPEED_BLOCK_SIZE,
            },
        },
    )

    metrics_all = []
    preds_all = []
    tuning_all = []

    for name, split_info in protocols.items():
        print("\n" + "=" * 80)
        print(f"Running protocol: {name} | {split_info.get('description', '')}")
        print("=" * 80)
        mdf, pdf, tdf = _run_one_protocol(name, split_info, data_path, seed, X)
        metrics_all.append(mdf)
        preds_all.append(pdf)
        tuning_all.append(tdf)

    df_metrics = _concat_nonempty_frames(metrics_all)
    df_preds = _concat_nonempty_frames(preds_all)
    df_tuning = _concat_nonempty_frames(tuning_all)
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
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
