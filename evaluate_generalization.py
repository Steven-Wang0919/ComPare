# -*- coding: utf-8 -*-
"""
evaluate_generalization.py

Generalization evaluation across multiple protocols with robustness summaries.
"""

import argparse
import os

import numpy as np
import pandas as pd

from common_utils import (
    build_protocol_splits,
    build_sample_tracking_columns,
    get_stratify_metadata,
    load_data_with_metadata,
)
from fair_tuning import build_protocol_aligned_inner_splits
from robustness_utils import (
    DEFAULT_OUTER_REPEATS,
    DEFAULT_STATS_METHOD,
    FIXED_HOLDOUT_OUTER_REPEAT_ID,
    FIXED_HOLDOUT_SPLIT_SEED,
    build_pairwise_stats,
    build_protocol_summary_wide,
    build_replicate_record,
    canonical_replicate_rule,
    is_canonical_replicate,
    normalize_training_seeds,
    split_seed_for_outer_repeat,
    summarize_replicate_metrics,
)
from run_utils import (
    append_manifest_outputs,
    build_multi_fold_split_artifact_payload,
    create_run_dir,
    save_dataframe,
    write_manifest,
)
from train_grnn import train_and_eval_grnn
from train_kan import train_and_eval_kan
from train_mlp import train_and_eval_mlp


SPEED_BLOCK_SIZE = 5
MODEL_NAMES = ("MLP", "GRNN", "KAN")
CANONICAL_RULE = canonical_replicate_rule()


def _artifact_source_files():
    base_dir = os.path.dirname(__file__)
    return [
        __file__,
        os.path.join(base_dir, "train_mlp.py"),
        os.path.join(base_dir, "train_grnn.py"),
        os.path.join(base_dir, "train_kan.py"),
        os.path.join(base_dir, "common_utils.py"),
        os.path.join(base_dir, "fair_tuning.py"),
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


def _run_model(model_name, data_path, train_seed, split_indices, inner_tuning_kwargs, *, artifact_dir, artifact_extra, save_artifacts):
    common_kwargs = dict(
        data_path=data_path,
        split_indices=split_indices,
        save_csv_path=None,
        save_artifacts=save_artifacts,
        artifact_dir=artifact_dir if save_artifacts else None,
        save_test_slice=save_artifacts,
        artifact_extra=artifact_extra,
        **inner_tuning_kwargs,
    )
    if model_name == "MLP":
        return train_and_eval_mlp(random_state=train_seed, **common_kwargs)
    if model_name == "GRNN":
        return train_and_eval_grnn(random_state=train_seed, **common_kwargs)
    if model_name == "KAN":
        return train_and_eval_kan(seed=train_seed, **common_kwargs)
    raise ValueError(f"Unsupported model_name: {model_name}")


def _append_replicate_meta(df, replicate):
    out = df.copy()
    out["protocol"] = replicate["protocol"]
    out["protocol_family"] = replicate["protocol_family"]
    out["fold_id"] = int(replicate["fold_id"])
    out["train_seed"] = int(replicate["train_seed"])
    out["outer_repeat_id"] = int(replicate["outer_repeat_id"])
    out["split_seed"] = int(replicate["split_seed"])
    out["replicate_id"] = replicate["replicate_id"]
    out["is_canonical_replicate"] = 1 if replicate["is_canonical_replicate"] else 0
    out["canonical_replicate_rule"] = CANONICAL_RULE
    return out


def _run_one_protocol(replicate, split_info, data_path, X, sample_meta, run_dir, pred_relpath):
    split_indices = (
        split_info["idx_train"],
        split_info["idx_val"],
        split_info["idx_test"],
    )
    inner_split_strategy, inner_tuning_kwargs = _build_inner_tuning_spec(X, split_info)
    save_artifacts = bool(replicate["is_canonical_replicate"])

    results = {}
    for model_name in MODEL_NAMES:
        results[model_name] = _run_model(
            model_name,
            data_path,
            int(replicate["train_seed"]),
            split_indices,
            inner_tuning_kwargs,
            artifact_dir=os.path.join(run_dir, "artifacts", replicate["protocol"], model_name),
            artifact_extra={
                "run_dir": run_dir.replace("\\", "/"),
                "protocol": replicate["protocol"],
                "protocol_desc": split_info.get("description", ""),
                "train_seed": int(replicate["train_seed"]),
                "outer_repeat_id": int(replicate["outer_repeat_id"]),
                "split_seed": int(replicate["split_seed"]),
                "replicate_id": replicate["replicate_id"],
                "is_canonical_replicate": int(replicate["is_canonical_replicate"]),
                "canonical_replicate_rule": CANONICAL_RULE,
                "reference_output": {
                    "path": pred_relpath,
                    "filter_column": "protocol",
                    "filter_value": replicate["protocol"],
                    "prediction_column": f"{model_name}_pred",
                    "target_column": "true",
                },
            },
            save_artifacts=save_artifacts,
        )

    ref_model = next(iter(results.keys()))
    ref_res = results[ref_model]

    pred_df = pd.DataFrame({
        "protocol": replicate["protocol"],
        **build_sample_tracking_columns(sample_meta, split_info["idx_test"]),
        "opening": ref_res["x_test_raw"][:, 0],
        "speed": ref_res["x_test_raw"][:, 1],
        "true": ref_res["y_true"],
    })
    for model_name, res in results.items():
        pred_df[f"{model_name}_pred"] = res["y_pred"]
    pred_df = _append_replicate_meta(pred_df, replicate)

    metrics_rows = []
    tuning_rows = []
    for model_name, res in results.items():
        row = {
            "protocol_desc": split_info.get("description", ""),
            "model": model_name,
            "r2": res["r2"],
            "are": res["are"],
            "train_size": len(split_info["idx_train"]),
            "val_size": len(split_info["idx_val"]),
            "test_size": len(split_info["idx_test"]),
            "n_test": len(split_info["idx_test"]),
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
                "protocol": replicate["protocol"],
                "protocol_family": replicate["protocol_family"],
                "fold_id": int(replicate["fold_id"]),
                "train_seed": int(replicate["train_seed"]),
                "outer_repeat_id": int(replicate["outer_repeat_id"]),
                "split_seed": int(replicate["split_seed"]),
                "replicate_id": replicate["replicate_id"],
                "protocol_desc": split_info.get("description", ""),
                "model": model_name,
            }
            merged.update(tuning_row)
            tuning_rows.append(merged)

    metrics_df = _append_replicate_meta(pd.DataFrame(metrics_rows), replicate)

    artifact_outputs = []
    if save_artifacts:
        for res in results.values():
            artifact_outputs.extend(_artifact_outputs_from_result(res, run_dir))

    return metrics_df, pred_df, pd.DataFrame(tuning_rows), artifact_outputs


def _build_protocol_specs(X):
    specs = [{"name": "random_interp", "kind": "random"}]
    unique_openings = sorted(np.unique(X[:, 0]).tolist())
    for op in unique_openings:
        op_name = int(op) if float(op).is_integer() else op
        specs.append({
            "name": f"leave_opening_{op_name}_out",
            "kind": "fixed",
            "kwargs": {"protocol": "leave_one_opening_out", "holdout_opening": op},
        })
    for protocol_name, smin, smax in [
        ("leave_speed_20_24_out", 20, 24),
        ("leave_speed_38_42_out", 38, 42),
        ("leave_speed_56_60_out", 56, 60),
    ]:
        specs.append({
            "name": protocol_name,
            "kind": "fixed",
            "kwargs": {
                "protocol": "leave_speed_block_out",
                "holdout_speed_min": smin,
                "holdout_speed_max": smax,
            },
        })
    return specs


def _build_replicate_jobs(X, y, training_seeds, outer_repeats, primary_seed):
    protocol_specs = _build_protocol_specs(X)
    jobs = []
    split_payload_folds = []
    payload_fold_id = 1
    for protocol_index, spec in enumerate(protocol_specs, start=1):
        protocol_name = spec["name"]
        family = _protocol_family(protocol_name)
        if spec["kind"] == "random":
            for outer_repeat_id in range(1, int(outer_repeats) + 1):
                split_seed = split_seed_for_outer_repeat(outer_repeat_id)
                split_info = build_protocol_splits(
                    X,
                    y,
                    protocol="random_stratified",
                    random_state=split_seed,
                    stratify_view="forward",
                )
                for train_seed in training_seeds:
                    replicate = build_replicate_record(
                        protocol=protocol_name,
                        fold_id=protocol_index,
                        train_seed=train_seed,
                        outer_repeat_id=outer_repeat_id,
                        split_seed=split_seed,
                        extra={"protocol_family": family},
                    )
                    replicate["is_canonical_replicate"] = bool(
                        is_canonical_replicate(replicate, primary_seed=primary_seed)
                    )
                    jobs.append((replicate, split_info))
                    split_payload_folds.append({
                        "fold_id": payload_fold_id,
                        "protocol": protocol_name,
                        "idx_train": split_info["idx_train"],
                        "idx_val": split_info["idx_val"],
                        "idx_test": split_info["idx_test"],
                        "train_seed": replicate["train_seed"],
                        "outer_repeat_id": replicate["outer_repeat_id"],
                        "split_seed": replicate["split_seed"],
                        "replicate_id": replicate["replicate_id"],
                        **get_stratify_metadata("forward"),
                    })
                    payload_fold_id += 1
        else:
            for train_seed in training_seeds:
                split_info = build_protocol_splits(
                    X,
                    y,
                    random_state=int(train_seed),
                    stratify_view="forward",
                    **spec["kwargs"],
                )
                replicate = build_replicate_record(
                    protocol=protocol_name,
                    fold_id=protocol_index,
                    train_seed=train_seed,
                    outer_repeat_id=FIXED_HOLDOUT_OUTER_REPEAT_ID,
                    split_seed=FIXED_HOLDOUT_SPLIT_SEED,
                    extra={"protocol_family": family},
                )
                replicate["is_canonical_replicate"] = bool(
                    is_canonical_replicate(replicate, primary_seed=primary_seed)
                )
                jobs.append((replicate, split_info))
                split_payload_folds.append({
                    "fold_id": payload_fold_id,
                    "protocol": protocol_name,
                    "idx_train": split_info["idx_train"],
                    "idx_val": split_info["idx_val"],
                    "idx_test": split_info["idx_test"],
                    "train_seed": replicate["train_seed"],
                    "outer_repeat_id": replicate["outer_repeat_id"],
                    "split_seed": replicate["split_seed"],
                    "replicate_id": replicate["replicate_id"],
                    **get_stratify_metadata("forward"),
                })
                payload_fold_id += 1
    return jobs, split_payload_folds


def _make_protocol_family_summary(df_metrics):
    return summarize_replicate_metrics(
        df_metrics,
        group_cols=["protocol_family", "model"],
        metric_cols=["r2", "are", "n_test"],
        passthrough_cols=["inner_split_strategy"],
        include_min_max=True,
    ).sort_values(["protocol_family", "model"]).reset_index(drop=True)


def _make_opening_cv_summary(df_metrics):
    df = df_metrics[df_metrics["protocol"].str.startswith("leave_opening_")].copy()
    if len(df) == 0:
        return pd.DataFrame()
    return summarize_replicate_metrics(
        df,
        group_cols=["model"],
        metric_cols=["r2", "are", "n_test"],
        include_min_max=True,
    ).sort_values("r2", ascending=False).reset_index(drop=True)


def _make_speed_cv_summary(df_metrics):
    df = df_metrics[df_metrics["protocol"].str.startswith("leave_speed_")].copy()
    if len(df) == 0:
        return pd.DataFrame()
    return summarize_replicate_metrics(
        df,
        group_cols=["model"],
        metric_cols=["r2", "are", "n_test"],
        include_min_max=True,
    ).sort_values("r2", ascending=False).reset_index(drop=True)


def _make_pairwise_stats(df_metrics, primary_seed):
    pair_key_cols = ["protocol", "fold_id", "train_seed", "outer_repeat_id"]
    overall = df_metrics.copy()
    overall["analysis_scope"] = "all_protocols"
    df_overall = build_pairwise_stats(
        overall,
        metric_specs=[
            {"column": "r2", "higher_is_better": True},
            {"column": "are", "higher_is_better": False},
        ],
        model_col="model",
        pair_key_cols=pair_key_cols,
        analysis_group_cols=["analysis_scope"],
        stats_seed=int(primary_seed),
    )
    family = df_metrics.copy()
    family["analysis_scope"] = "protocol_family"
    df_family = build_pairwise_stats(
        family,
        metric_specs=[
            {"column": "r2", "higher_is_better": True},
            {"column": "are", "higher_is_better": False},
        ],
        model_col="model",
        pair_key_cols=pair_key_cols,
        analysis_group_cols=["analysis_scope", "protocol_family"],
        stats_seed=int(primary_seed),
    )
    return pd.concat([df_overall, df_family], ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generalization protocols with robustness summaries.")
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
    X, y, sample_meta = load_data_with_metadata(data_path)
    jobs, split_payload_folds = _build_replicate_jobs(
        X,
        y,
        training_seeds=training_seeds,
        outer_repeats=args.outer_repeats,
        primary_seed=primary_seed,
    )
    split_payload = build_multi_fold_split_artifact_payload(
        split_payload_folds,
        n_samples=len(X),
    )

    run_dir = create_run_dir("evaluate_generalization")
    write_manifest(
        run_dir,
        script_name="evaluate_generalization.py",
        data_path=data_path,
        seed=training_seeds,
        params={
            "models": list(MODEL_NAMES),
            "primary_seed": primary_seed,
            "training_seeds": training_seeds,
            "outer_repeats": int(args.outer_repeats),
            "stats_method": args.stats_method,
            "split_seed_policy": "split_seed = 1000 + outer_repeat_id; split_seed is orthogonal to train_seed",
            "paired_key_definition": {
                "random_protocol": "(protocol, fold_id, train_seed, outer_repeat_id)",
                "fixed_holdout": "(protocol, fold_id, train_seed)",
            },
            "canonical_replicate_rule": CANONICAL_RULE,
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
        source_files=_artifact_source_files(),
        split_payload=split_payload,
    )

    metrics_all = []
    preds_all = []
    tuning_all = []
    artifact_outputs = []
    pred_relpath = "protocol_predictions.csv"

    for replicate, split_info in jobs:
        mdf, pdf, tdf, protocol_artifacts = _run_one_protocol(
            replicate,
            split_info,
            data_path,
            X,
            sample_meta,
            run_dir,
            pred_relpath,
        )
        metrics_all.append(mdf)
        if replicate["is_canonical_replicate"]:
            preds_all.append(pdf)
        if tdf is not None and isinstance(tdf, pd.DataFrame) and not tdf.empty:
            tuning_all.append(tdf)
        artifact_outputs.extend(protocol_artifacts)

    df_metrics = pd.concat(metrics_all, ignore_index=True)
    df_preds = pd.concat(preds_all, ignore_index=True) if len(preds_all) > 0 else pd.DataFrame()
    df_tuning = pd.concat(tuning_all, ignore_index=True) if len(tuning_all) > 0 else pd.DataFrame()
    df_protocol_summary = build_protocol_summary_wide(df_metrics)
    df_family_summary = _make_protocol_family_summary(df_metrics)
    df_opening_cv_summary = _make_opening_cv_summary(df_metrics)
    df_speed_cv_summary = _make_speed_cv_summary(df_metrics)
    df_pairwise = _make_pairwise_stats(df_metrics, primary_seed)

    save_dataframe(df_metrics, os.path.join(run_dir, "protocol_metrics.csv"))
    save_dataframe(df_preds, os.path.join(run_dir, "protocol_predictions.csv"))
    save_dataframe(df_protocol_summary, os.path.join(run_dir, "protocol_summary.csv"))
    save_dataframe(df_family_summary, os.path.join(run_dir, "protocol_family_summary.csv"))
    save_dataframe(df_opening_cv_summary, os.path.join(run_dir, "opening_cv_summary.csv"))
    save_dataframe(df_speed_cv_summary, os.path.join(run_dir, "speed_cv_summary.csv"))
    save_dataframe(df_pairwise, os.path.join(run_dir, "protocol_pairwise_stats.csv"))
    if len(df_tuning) > 0:
        save_dataframe(df_tuning, os.path.join(run_dir, "protocol_tuning_audit.csv"))

    outputs = [
        {"path": "protocol_metrics.csv"},
        {"path": "protocol_predictions.csv"},
        {"path": "protocol_summary.csv"},
        {"path": "protocol_family_summary.csv"},
        {"path": "opening_cv_summary.csv"},
        {"path": "speed_cv_summary.csv"},
        {"path": "protocol_pairwise_stats.csv"},
    ]
    if len(df_tuning) > 0:
        outputs.append({"path": "protocol_tuning_audit.csv"})
    outputs.extend(artifact_outputs)

    append_manifest_outputs(run_dir, outputs)
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
