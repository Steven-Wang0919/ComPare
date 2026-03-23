# -*- coding: utf-8 -*-
"""
compare_all.py

Unified forward/inverse comparison with multi-replicate robustness reporting.
"""

import argparse
import os

import numpy as np
import pandas as pd

from common_utils import get_stratify_metadata, get_train_val_test_indices, load_data_with_metadata
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
    build_pairwise_stats,
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


FORWARD_MODELS = ("MLP", "GRNN", "KAN")
INVERSE_MODELS = ("inverse_MLP", "inverse_GRNN", "inverse_KAN")
CANONICAL_RULE = canonical_replicate_rule()


def _artifact_source_files():
    base_dir = os.path.dirname(__file__)
    return [
        __file__,
        os.path.join(base_dir, "train_mlp.py"),
        os.path.join(base_dir, "train_grnn.py"),
        os.path.join(base_dir, "train_kan.py"),
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


def _merge_summary_hyperparams(df_summary, df_replicates):
    if len(df_summary) == 0:
        return df_summary
    keep_cols = ["Model", "Hyperparams"]
    base = (
        df_replicates.sort_values(
            ["is_canonical_replicate", "train_seed", "outer_repeat_id"],
            ascending=[False, True, True],
        )[keep_cols]
        .drop_duplicates(subset=["Model"], keep="first")
    )
    return df_summary.merge(base, on="Model", how="left")


def _save_tuning_audit(output_dir, filename, tuning_rows):
    if len(tuning_rows) == 0:
        return None
    path = os.path.join(output_dir, filename)
    save_dataframe(pd.DataFrame(tuning_rows), path)
    return path


def _build_compare_replicates(X, y, training_seeds, outer_repeats, primary_seed):
    replicates = []
    split_payload_folds = []
    fold_id_counter = 1
    for outer_repeat_id in range(1, int(outer_repeats) + 1):
        split_seed = split_seed_for_outer_repeat(outer_repeat_id)
        split_indices = get_train_val_test_indices(
            X=X,
            y=y,
            random_state=split_seed,
            stratify_view="physical_joint",
        )
        for train_seed in training_seeds:
            replicate = build_replicate_record(
                protocol="random_interp",
                fold_id=1,
                train_seed=train_seed,
                outer_repeat_id=outer_repeat_id,
                split_seed=split_seed,
            )
            replicate["is_canonical_replicate"] = bool(
                is_canonical_replicate(replicate, primary_seed=primary_seed)
            )
            replicate["split_indices"] = split_indices
            replicates.append(replicate)
            split_payload_folds.append({
                "fold_id": fold_id_counter,
                "protocol": replicate["protocol"],
                "idx_train": split_indices[0],
                "idx_val": split_indices[1],
                "idx_test": split_indices[2],
                "train_seed": replicate["train_seed"],
                "outer_repeat_id": replicate["outer_repeat_id"],
                "split_seed": replicate["split_seed"],
                "replicate_id": replicate["replicate_id"],
                **get_stratify_metadata("physical_joint"),
            })
            fold_id_counter += 1
    return replicates, split_payload_folds


def _run_forward_once(output_dir, data_path, replicate):
    split_indices = replicate["split_indices"]
    save_artifacts = bool(replicate["is_canonical_replicate"])
    pred_relpath = "forward_model_predictions.csv"
    artifact_base = os.path.join(output_dir, "artifacts", "forward")

    common_meta = {
        "run_dir": output_dir.replace("\\", "/"),
        "protocol": replicate["protocol"],
        "fold_id": int(replicate["fold_id"]),
        "train_seed": int(replicate["train_seed"]),
        "outer_repeat_id": int(replicate["outer_repeat_id"]),
        "split_seed": int(replicate["split_seed"]),
        "replicate_id": replicate["replicate_id"],
        "is_canonical_replicate": int(replicate["is_canonical_replicate"]),
        "canonical_replicate_rule": CANONICAL_RULE,
    }
    mlp_res = train_and_eval_mlp(
        data_path=data_path,
        random_state=int(replicate["train_seed"]),
        save_csv_path=None,
        split_indices=split_indices,
        outer_split_stratify_view="physical_joint",
        save_artifacts=save_artifacts,
        artifact_dir=os.path.join(artifact_base, "MLP") if save_artifacts else None,
        save_test_slice=save_artifacts,
        artifact_extra={
            **common_meta,
            "reference_output": {
                "path": pred_relpath,
                "prediction_column": "MLP_pred",
                "target_column": "true",
            },
        },
    )
    grnn_res = train_and_eval_grnn(
        data_path=data_path,
        save_csv_path=None,
        random_state=int(replicate["train_seed"]),
        split_indices=split_indices,
        outer_split_stratify_view="physical_joint",
        save_artifacts=save_artifacts,
        artifact_dir=os.path.join(artifact_base, "GRNN") if save_artifacts else None,
        save_test_slice=save_artifacts,
        artifact_extra={
            **common_meta,
            "reference_output": {
                "path": pred_relpath,
                "prediction_column": "GRNN_pred",
                "target_column": "true",
            },
        },
    )
    kan_res = train_and_eval_kan(
        data_path=data_path,
        seed=int(replicate["train_seed"]),
        save_csv_path=None,
        split_indices=split_indices,
        outer_split_stratify_view="physical_joint",
        save_artifacts=save_artifacts,
        artifact_dir=os.path.join(artifact_base, "KAN") if save_artifacts else None,
        save_test_slice=save_artifacts,
        artifact_extra={
            **common_meta,
            "reference_output": {
                "path": pred_relpath,
                "prediction_column": "KAN_pred",
                "target_column": "true",
            },
        },
    )

    y_true_mlp = _to_1d_array(mlp_res["y_true"], "MLP y_true")
    y_true_grnn = _to_1d_array(grnn_res["y_true"], "GRNN y_true")
    y_true_kan = _to_1d_array(kan_res["y_true"], "KAN y_true")
    y_pred_mlp = _to_1d_array(mlp_res["y_pred"], "MLP y_pred")
    y_pred_grnn = _to_1d_array(grnn_res["y_pred"], "GRNN y_pred")
    y_pred_kan = _to_1d_array(kan_res["y_pred"], "KAN y_pred")

    _validate_same_values(y_true_mlp, y_true_grnn, "MLP y_true", "GRNN y_true")
    _validate_same_values(y_true_mlp, y_true_kan, "MLP y_true", "KAN y_true")
    _validate_same_values(mlp_res["test_sample_id"], grnn_res["test_sample_id"], "MLP sample_id", "GRNN sample_id")
    _validate_same_values(mlp_res["test_sample_id"], kan_res["test_sample_id"], "MLP sample_id", "KAN sample_id")

    metrics_rows = _append_replicate_meta(pd.DataFrame([
        {
            "Task": "forward",
            "Model": "MLP",
            "DisplayName": "MLP",
            "ArchitectureNote": "",
            "R2": mlp_res["r2"],
            "ARE(%)": mlp_res["are"],
            "n_test": len(y_true_mlp),
            "Hyperparams": f"hidden={mlp_res.get('best_hidden')}, alpha={mlp_res.get('best_alpha')}",
        },
        {
            "Task": "forward",
            "Model": "GRNN",
            "DisplayName": "GRNN",
            "ArchitectureNote": "",
            "R2": grnn_res["r2"],
            "ARE(%)": grnn_res["are"],
            "n_test": len(y_true_mlp),
            "Hyperparams": f"sigma={_fmt_float(grnn_res.get('best_sigma'))}",
        },
        {
            "Task": "forward",
            "Model": "KAN",
            "DisplayName": "KAN",
            "ArchitectureNote": "forward spline/grid KAN",
            "R2": kan_res["r2"],
            "ARE(%)": kan_res["are"],
            "n_test": len(y_true_mlp),
            "Hyperparams": (
                f"hidden={kan_res.get('best_hidden_dim')}, "
                f"lr={_fmt_float(kan_res.get('best_lr'))}, "
                f"wd={_fmt_float(kan_res.get('best_weight_decay'))}"
            ),
        },
    ]), replicate)

    pred_df = _append_replicate_meta(pd.DataFrame({
        "sample_id": np.asarray(mlp_res["test_sample_id"], dtype=int),
        "source_row_number": np.asarray(mlp_res["test_source_row_number"], dtype=int),
        "true": y_true_mlp,
        "MLP_pred": y_pred_mlp,
        "GRNN_pred": y_pred_grnn,
        "KAN_pred": y_pred_kan,
    }), replicate)

    tuning_rows = []
    for model_name, result in {"MLP": mlp_res, "GRNN": grnn_res, "KAN": kan_res}.items():
        for row in result.get("tuning_records", []):
            merged = dict(common_meta)
            merged["comparison_scope"] = "forward"
            merged["reported_model"] = model_name
            merged.update(row)
            tuning_rows.append(merged)

    artifact_outputs = []
    if save_artifacts:
        artifact_outputs.extend(_artifact_outputs_from_result(mlp_res, output_dir))
        artifact_outputs.extend(_artifact_outputs_from_result(grnn_res, output_dir))
        artifact_outputs.extend(_artifact_outputs_from_result(kan_res, output_dir))

    return {
        "metrics_df": metrics_rows,
        "pred_df": pred_df,
        "tuning_rows": tuning_rows,
        "artifact_outputs": artifact_outputs,
    }


def _run_inverse_once(output_dir, data_path, replicate):
    split_indices = replicate["split_indices"]
    save_artifacts = bool(replicate["is_canonical_replicate"])
    pred_relpath = "inverse_model_predictions_all.csv"
    artifact_base = os.path.join(output_dir, "artifacts", "inverse")

    common_meta = {
        "run_dir": output_dir.replace("\\", "/"),
        "protocol": replicate["protocol"],
        "fold_id": int(replicate["fold_id"]),
        "train_seed": int(replicate["train_seed"]),
        "outer_repeat_id": int(replicate["outer_repeat_id"]),
        "split_seed": int(replicate["split_seed"]),
        "replicate_id": replicate["replicate_id"],
        "is_canonical_replicate": int(replicate["is_canonical_replicate"]),
        "canonical_replicate_rule": CANONICAL_RULE,
    }
    mlp_res = train_and_eval_inverse_mlp(
        data_path=data_path,
        random_state=int(replicate["train_seed"]),
        save_outputs_dir=None,
        split_indices=split_indices,
        outer_split_stratify_view="physical_joint",
        save_artifacts=save_artifacts,
        artifact_dir=os.path.join(artifact_base, "inverse_MLP") if save_artifacts else None,
        save_test_slice=save_artifacts,
        artifact_extra={
            **common_meta,
            "reference_output": {
                "path": pred_relpath,
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
        outer_split_stratify_view="physical_joint",
        save_artifacts=save_artifacts,
        artifact_dir=os.path.join(artifact_base, "inverse_GRNN") if save_artifacts else None,
        save_test_slice=save_artifacts,
        artifact_extra={
            **common_meta,
            "reference_output": {
                "path": pred_relpath,
                "prediction_column": "inverse_GRNN_pred",
                "target_column": "true_speed_r_min",
            },
        },
    )
    kan_res = train_and_eval_inverse_kan_v2(
        data_path=data_path,
        seed=int(replicate["train_seed"]),
        split_indices=split_indices,
        outer_split_stratify_view="physical_joint",
        save_artifacts=save_artifacts,
        artifact_dir=os.path.join(artifact_base, "inverse_KAN") if save_artifacts else None,
        save_test_slice=save_artifacts,
        artifact_extra={
            **common_meta,
            "reference_output": {
                "path": pred_relpath,
                "prediction_column": "inverse_KAN_pred",
                "target_column": "true_speed_r_min",
            },
        },
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
    _validate_same_values(strat_open_all_mlp, strat_open_all_grnn, "inverse_MLP strategy_opening_all", "inverse_GRNN strategy_opening_all")
    _validate_same_values(strat_open_all_mlp, strat_open_all_kan, "inverse_MLP strategy_opening_all", "inverse_KAN strategy_opening_all")
    _validate_same_values(mlp_res["sample_id_all"], grnn_res["sample_id_all"], "inverse_MLP sample_id_all", "inverse_GRNN sample_id_all")
    _validate_same_values(mlp_res["sample_id_all"], kan_res["sample_id_all"], "inverse_MLP sample_id_all", "inverse_KAN sample_id_all")
    _validate_same_mask(policy_mask_mlp, policy_mask_grnn, "inverse_MLP policy_mask", "inverse_GRNN policy_mask")
    _validate_same_mask(policy_mask_mlp, policy_mask_kan, "inverse_MLP policy_mask", "inverse_KAN policy_mask")

    metrics_df = _append_replicate_meta(pd.DataFrame([
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
            "Hyperparams": f"hidden={mlp_res.get('best_hidden')}, alpha={mlp_res.get('best_alpha')}",
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
    ]), replicate)

    df_all = _append_replicate_meta(pd.DataFrame({
        "sample_id": np.asarray(mlp_res["sample_id_all"], dtype=int),
        "source_row_number": np.asarray(mlp_res["source_row_number_all"], dtype=int),
        "target_mass_g_min": mass_all_mlp,
        "actual_opening_mm": opening_all_mlp,
        "strategy_opening_mm": strat_open_all_mlp,
        "true_speed_r_min": y_true_all_mlp,
        "inverse_MLP_pred": y_pred_all_mlp,
        "inverse_GRNN_pred": y_pred_all_grnn,
        "inverse_KAN_pred": y_pred_all_kan,
        "policy_match": policy_mask_mlp.astype(int),
    }), replicate)
    main_idx = np.where(policy_mask_mlp)[0]
    df_main = _append_replicate_meta(pd.DataFrame({
        "sample_id": np.asarray(mlp_res["sample_id_all"], dtype=int)[main_idx],
        "source_row_number": np.asarray(mlp_res["source_row_number_all"], dtype=int)[main_idx],
        "target_mass_g_min": mass_all_mlp[main_idx],
        "actual_opening_mm": opening_all_mlp[main_idx],
        "strategy_opening_mm": strat_open_all_mlp[main_idx],
        "true_speed_r_min": y_true_all_mlp[main_idx],
        "inverse_MLP_pred": y_pred_all_mlp[main_idx],
        "inverse_GRNN_pred": y_pred_all_grnn[main_idx],
        "inverse_KAN_pred": y_pred_all_kan[main_idx],
    }), replicate)

    tuning_rows = []
    for model_name, result in {
        "inverse_MLP": mlp_res,
        "inverse_GRNN": grnn_res,
        "inverse_KAN": kan_res,
    }.items():
        for row in result.get("tuning_records", []):
            merged = dict(common_meta)
            merged["comparison_scope"] = "inverse"
            merged["reported_model"] = model_name
            merged.update(row)
            tuning_rows.append(merged)

    artifact_outputs = []
    if save_artifacts:
        artifact_outputs.extend(_artifact_outputs_from_result(mlp_res, output_dir))
        artifact_outputs.extend(_artifact_outputs_from_result(grnn_res, output_dir))
        artifact_outputs.extend(_artifact_outputs_from_result(kan_res, output_dir))

    return {
        "metrics_df": metrics_df,
        "all_pred_df": df_all,
        "main_pred_df": df_main,
        "tuning_rows": tuning_rows,
        "artifact_outputs": artifact_outputs,
    }


def run_forward_compare(output_dir, data_path, replicates, primary_seed):
    metrics_all = []
    tuning_rows = []
    artifact_outputs = []
    canonical_pred_df = None

    for replicate in replicates:
        result = _run_forward_once(output_dir, data_path, replicate)
        metrics_all.append(result["metrics_df"])
        tuning_rows.extend(result["tuning_rows"])
        artifact_outputs.extend(result["artifact_outputs"])
        if replicate["is_canonical_replicate"]:
            canonical_pred_df = result["pred_df"]

    df_replicates = pd.concat(metrics_all, ignore_index=True)
    df_summary = summarize_replicate_metrics(
        df_replicates,
        group_cols=["Task", "Model", "DisplayName", "ArchitectureNote"],
        metric_cols=["R2", "ARE(%)", "n_test"],
    )
    df_summary = _merge_summary_hyperparams(df_summary, df_replicates)
    df_summary = df_summary.sort_values(
        "Model",
        key=lambda s: s.map({name: idx for idx, name in enumerate(FORWARD_MODELS)}),
    ).reset_index(drop=True)

    df_pairwise = build_pairwise_stats(
        df_replicates,
        metric_specs=[
            {"column": "R2", "higher_is_better": True},
            {"column": "ARE(%)", "higher_is_better": False},
        ],
        model_col="Model",
        pair_key_cols=["protocol", "fold_id", "train_seed", "outer_repeat_id"],
        stats_seed=int(primary_seed),
    )

    replicate_path = os.path.join(output_dir, "forward_model_replicate_metrics.csv")
    metrics_path = os.path.join(output_dir, "forward_model_metrics.csv")
    pred_path = os.path.join(output_dir, "forward_model_predictions.csv")
    pairwise_path = os.path.join(output_dir, "forward_model_pairwise_stats.csv")
    tuning_audit_path = _save_tuning_audit(output_dir, "forward_tuning_audit.csv", tuning_rows)

    save_dataframe(df_replicates, replicate_path)
    save_dataframe(df_summary, metrics_path)
    if canonical_pred_df is not None:
        save_dataframe(canonical_pred_df, pred_path)
    save_dataframe(df_pairwise, pairwise_path)

    return {
        "metrics_path": metrics_path,
        "replicate_metrics_path": replicate_path,
        "pred_path": pred_path if canonical_pred_df is not None else None,
        "pairwise_path": pairwise_path,
        "tuning_audit_path": tuning_audit_path,
        "artifact_outputs": artifact_outputs,
    }


def run_inverse_compare(output_dir, data_path, replicates, primary_seed):
    metrics_all = []
    tuning_rows = []
    artifact_outputs = []
    canonical_all_df = None
    canonical_main_df = None

    for replicate in replicates:
        result = _run_inverse_once(output_dir, data_path, replicate)
        metrics_all.append(result["metrics_df"])
        tuning_rows.extend(result["tuning_rows"])
        artifact_outputs.extend(result["artifact_outputs"])
        if replicate["is_canonical_replicate"]:
            canonical_all_df = result["all_pred_df"]
            canonical_main_df = result["main_pred_df"]

    df_replicates = pd.concat(metrics_all, ignore_index=True)
    df_summary = summarize_replicate_metrics(
        df_replicates,
        group_cols=["Task", "Model", "DisplayName", "ArchitectureNote"],
        metric_cols=["R2_main", "ARE_main(%)", "R2_all", "ARE_all(%)", "n_main", "main_ratio", "n_all"],
    )
    df_summary = _merge_summary_hyperparams(df_summary, df_replicates)
    df_summary = df_summary.sort_values(
        "Model",
        key=lambda s: s.map({name: idx for idx, name in enumerate(INVERSE_MODELS)}),
    ).reset_index(drop=True)

    df_pairwise = build_pairwise_stats(
        df_replicates,
        metric_specs=[
            {"column": "R2_main", "higher_is_better": True},
            {"column": "ARE_main(%)", "higher_is_better": False},
            {"column": "R2_all", "higher_is_better": True},
            {"column": "ARE_all(%)", "higher_is_better": False},
        ],
        model_col="Model",
        pair_key_cols=["protocol", "fold_id", "train_seed", "outer_repeat_id"],
        stats_seed=int(primary_seed),
    )

    replicate_path = os.path.join(output_dir, "inverse_model_replicate_metrics.csv")
    metrics_path = os.path.join(output_dir, "inverse_model_metrics.csv")
    all_path = os.path.join(output_dir, "inverse_model_predictions_all.csv")
    main_path = os.path.join(output_dir, "inverse_model_predictions_main.csv")
    pairwise_path = os.path.join(output_dir, "inverse_model_pairwise_stats.csv")
    tuning_audit_path = _save_tuning_audit(output_dir, "inverse_tuning_audit.csv", tuning_rows)

    save_dataframe(df_replicates, replicate_path)
    save_dataframe(df_summary, metrics_path)
    if canonical_all_df is not None:
        save_dataframe(canonical_all_df, all_path)
    if canonical_main_df is not None:
        save_dataframe(canonical_main_df, main_path)
    save_dataframe(df_pairwise, pairwise_path)

    return {
        "metrics_path": metrics_path,
        "replicate_metrics_path": replicate_path,
        "all_path": all_path if canonical_all_df is not None else None,
        "main_path": main_path if canonical_main_df is not None else None,
        "pairwise_path": pairwise_path,
        "tuning_audit_path": tuning_audit_path,
        "artifact_outputs": artifact_outputs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run forward/inverse comparisons with robustness summaries.")
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
    X, y, _ = load_data_with_metadata(data_path)
    replicates, split_payload_folds = _build_compare_replicates(
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

    run_dir = create_run_dir("compare_all")
    manifest_path = write_manifest(
        run_dir,
        script_name="compare_all.py",
        data_path=data_path,
        seed=training_seeds,
        params={
            "forward_models": list(FORWARD_MODELS),
            "inverse_models": list(INVERSE_MODELS),
            "save_root": "runs",
            "primary_seed": primary_seed,
            "training_seeds": training_seeds,
            "outer_repeats": int(args.outer_repeats),
            "stats_method": args.stats_method,
            "split_seed_policy": "split_seed = 1000 + outer_repeat_id; split_seed is orthogonal to train_seed",
            "paired_key_definition": "(protocol, fold_id, train_seed, outer_repeat_id)",
            "canonical_replicate_rule": CANONICAL_RULE,
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
        source_files=_artifact_source_files(),
        split_payload=split_payload,
    )

    print(f"\nRun directory: {run_dir}")
    print(f"Manifest: {manifest_path}")

    forward_outputs = run_forward_compare(run_dir, data_path, replicates, primary_seed)
    inverse_outputs = run_inverse_compare(run_dir, data_path, replicates, primary_seed)

    manifest_outputs = [
        {"path": os.path.relpath(forward_outputs["metrics_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(forward_outputs["replicate_metrics_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(forward_outputs["pairwise_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(inverse_outputs["metrics_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(inverse_outputs["replicate_metrics_path"], run_dir).replace("\\", "/")},
        {"path": os.path.relpath(inverse_outputs["pairwise_path"], run_dir).replace("\\", "/")},
        *forward_outputs["artifact_outputs"],
        *inverse_outputs["artifact_outputs"],
    ]
    if forward_outputs["pred_path"] is not None:
        manifest_outputs.append({"path": os.path.relpath(forward_outputs["pred_path"], run_dir).replace("\\", "/")})
    if forward_outputs["tuning_audit_path"] is not None:
        manifest_outputs.append({"path": os.path.relpath(forward_outputs["tuning_audit_path"], run_dir).replace("\\", "/")})
    if inverse_outputs["all_path"] is not None:
        manifest_outputs.append({"path": os.path.relpath(inverse_outputs["all_path"], run_dir).replace("\\", "/")})
    if inverse_outputs["main_path"] is not None:
        manifest_outputs.append({"path": os.path.relpath(inverse_outputs["main_path"], run_dir).replace("\\", "/")})
    if inverse_outputs["tuning_audit_path"] is not None:
        manifest_outputs.append({"path": os.path.relpath(inverse_outputs["tuning_audit_path"], run_dir).replace("\\", "/")})

    append_manifest_outputs(run_dir, manifest_outputs)

    print("\nAll outputs saved to:")
    print(run_dir)


if __name__ == "__main__":
    main()
