# -*- coding: utf-8 -*-
"""
train_mlp.py

Forward MLP with a shared fair tuning protocol and replayable artifact bundles.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from common_utils import (
    average_relative_error,
    build_sample_tracking_columns,
    combine_train_val_indices,
    get_train_val_test_indices,
    load_data_with_metadata,
    validate_predefined_split_indices,
)
from fair_tuning import (
    ensure_fair_tuning_config,
    infer_inner_val_ratio,
    prepare_inner_cv,
    run_fair_tuning,
    tuning_config_to_dict,
)
from run_utils import (
    append_manifest_outputs,
    build_artifact_metadata,
    build_single_split_artifact_payload,
    build_split_indices_payload,
    build_tuning_protocol_payload,
    create_run_dir,
    ensure_dir,
    save_dataframe,
    save_test_slice,
    update_manifest_split_artifact,
    write_json,
    write_manifest,
)


EPS = 1e-8
DEFAULT_HIDDEN_LAYER_CANDIDATES = [(8,), (16,), (32,), (16, 16), (32, 16), (32, 32)]
DEFAULT_ALPHA_CANDIDATES = [1e-6, 1e-5, 1e-4, 1e-3]
MODEL_FILENAME = "model.joblib"
META_FILENAME = "meta.json"


def _artifact_source_files():
    base_dir = os.path.dirname(__file__)
    return [
        __file__,
        os.path.join(base_dir, "common_utils.py"),
        os.path.join(base_dir, "fair_tuning.py"),
        os.path.join(base_dir, "run_utils.py"),
    ]


def _build_candidate_configs(hidden_layer_candidates=None, alpha_candidates=None):
    hidden_layer_candidates = hidden_layer_candidates or DEFAULT_HIDDEN_LAYER_CANDIDATES
    alpha_candidates = alpha_candidates or DEFAULT_ALPHA_CANDIDATES
    return [
        {
            "hidden_layer_sizes": tuple(hidden),
            "alpha": float(alpha),
        }
        for hidden in hidden_layer_candidates
        for alpha in alpha_candidates
    ]


def _fit_predict_forward_mlp(
    X_train_raw,
    y_train_raw,
    X_eval_raw,
    *,
    hidden_layer_sizes,
    alpha,
    max_iter,
    random_state,
):
    X_min = X_train_raw.min(axis=0, keepdims=True)
    X_max = X_train_raw.max(axis=0, keepdims=True)
    y_min = float(np.min(y_train_raw))
    y_max = float(np.max(y_train_raw))

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + EPS)

    def norm_y(v):
        return (v - y_min) / (y_max - y_min + EPS)

    def denorm_y(v):
        return v * (y_max - y_min + EPS) + y_min

    model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        alpha=float(alpha),
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(norm_x(X_train_raw), norm_y(y_train_raw))
    y_pred_eval = denorm_y(model.predict(norm_x(X_eval_raw)))

    return {
        "model": model,
        "y_pred_eval": y_pred_eval,
        "norm_stats": {
            "X_min": X_min,
            "X_max": X_max,
            "y_min": y_min,
            "y_max": y_max,
        },
    }


def _save_forward_mlp_artifacts(
    *,
    model,
    artifact_dir,
    data_path,
    best_config,
    norm_stats,
    idx_tr,
    idx_val,
    idx_te,
    tuning_config,
    tuning_result,
    resolved_inner_splits=None,
    X_dev_raw,
    y_dev_raw,
    X_test_raw,
    y_test_raw,
    artifact_extra=None,
    artifact_source_files=None,
    model_filename=MODEL_FILENAME,
    meta_filename=META_FILENAME,
    save_test_slice_flag=False,
):
    ensure_dir(artifact_dir)
    model_path = os.path.join(artifact_dir, model_filename)
    meta_path = os.path.join(artifact_dir, meta_filename)
    test_inputs_path = None
    test_targets_path = None

    joblib.dump(model, model_path)
    if save_test_slice_flag:
        test_inputs_path, test_targets_path = save_test_slice(artifact_dir, X_test_raw, y_test_raw)

    tuning_payload = build_tuning_protocol_payload(
        tuning_config_to_dict(tuning_config),
        inner_split_strategy=tuning_result["inner_split_strategy"],
        inner_split_meta=tuning_result["inner_split_meta"],
        inner_splits=resolved_inner_splits,
        tuning_seed=int(tuning_config.seed),
        n_repeats=int(tuning_result["inner_fold_count"]),
        inner_val_ratio=tuning_config.inner_val_ratio,
    )
    meta = build_artifact_metadata(
        artifact_type="model_bundle",
        task_name="forward",
        model_name="MLP",
        model_class="sklearn.neural_network.MLPRegressor",
        data_path=data_path,
        best_config=best_config,
        normalization_params=norm_stats,
        split_indices=build_split_indices_payload(idx_tr, idx_val, idx_te),
        tuning_protocol=tuning_payload,
        training_domain={
            "feature_names": ["opening_mm", "speed_r_min"],
            "target_name": "mass_g_min",
            "opening_min": float(X_dev_raw[:, 0].min()),
            "opening_max": float(X_dev_raw[:, 0].max()),
            "speed_min": float(X_dev_raw[:, 1].min()),
            "speed_max": float(X_dev_raw[:, 1].max()),
            "mass_min": float(y_dev_raw.min()),
            "mass_max": float(y_dev_raw.max()),
        },
        extra={
            "model_file": model_filename,
            "meta_file": meta_filename,
            "test_inputs_file": os.path.basename(test_inputs_path) if test_inputs_path is not None else None,
            "test_targets_file": os.path.basename(test_targets_path) if test_targets_path is not None else None,
            **dict(artifact_extra or {}),
        },
        source_files=artifact_source_files or _artifact_source_files(),
    )
    write_json(meta_path, meta)
    return model_path, meta_path, test_inputs_path, test_targets_path


def train_and_eval_mlp(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=42,
    save_csv_path=None,
    split_indices=None,
    tuning_config=None,
    save_tuning_records_path=None,
    inner_splits=None,
    inner_split_strategy=None,
    inner_split_meta=None,
    save_artifacts=False,
    artifact_dir=None,
    model_filename=MODEL_FILENAME,
    meta_filename=META_FILENAME,
    save_test_slice=False,
    artifact_extra=None,
    artifact_source_files=None,
):
    print("\n=== 训练 MLP（公平调参协议） ===")

    X, y, sample_meta = load_data_with_metadata(data_path)
    if split_indices is None:
        idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X, y=y, random_state=random_state)
    else:
        idx_tr, idx_val, idx_te = validate_predefined_split_indices(
            len(X), split_indices[0], split_indices[1], split_indices[2]
        )

    inner_val_ratio = None if inner_splits is not None else infer_inner_val_ratio(idx_tr, idx_val)
    tuning_config = ensure_fair_tuning_config(
        tuning_config,
        seed=random_state,
        inner_val_ratio=inner_val_ratio,
    )

    dev_idx = combine_train_val_indices(idx_tr, idx_val)
    X_dev_raw = X[dev_idx]
    y_dev_raw = y[dev_idx]
    X_test_raw = X[idx_te]
    y_test_raw = y[idx_te]
    test_tracking = build_sample_tracking_columns(sample_meta, idx_te)
    split_indices_payload = build_single_split_artifact_payload(
        idx_tr,
        idx_val,
        idx_te,
        n_samples=len(X),
    )

    inner_splits, inner_split_strategy, inner_split_meta = prepare_inner_cv(
        X_dev_raw,
        y_dev_raw,
        tuning_config,
        inner_splits=inner_splits,
        inner_split_strategy=inner_split_strategy,
        inner_split_meta=inner_split_meta,
    )
    candidate_configs = _build_candidate_configs(hidden_layer_candidates, alpha_candidates)

    def eval_candidate_fn(*, config, idx_train, idx_val, fold_id, split_meta):
        del split_meta
        res = _fit_predict_forward_mlp(
            X_dev_raw[idx_train],
            y_dev_raw[idx_train],
            X_dev_raw[idx_val],
            hidden_layer_sizes=config["hidden_layer_sizes"],
            alpha=config["alpha"],
            max_iter=max_iter,
            random_state=int(tuning_config.seed) + int(fold_id),
        )
        y_pred_val = res["y_pred_eval"]
        y_true_val = y_dev_raw[idx_val]
        return {
            "val_r2": float(r2_score(y_true_val, y_pred_val)),
            "val_are": float(average_relative_error(y_true_val, y_pred_val)),
        }

    tuning_result = run_fair_tuning(
        candidate_configs=candidate_configs,
        inner_splits=inner_splits,
        eval_candidate_fn=eval_candidate_fn,
        tuning_config=tuning_config,
        model_name="MLP",
        task_name="forward",
        inner_split_strategy=inner_split_strategy,
        inner_split_meta=inner_split_meta,
    )
    best_config = tuning_result["best_config"]

    final_fit = _fit_predict_forward_mlp(
        X_dev_raw,
        y_dev_raw,
        X_test_raw,
        hidden_layer_sizes=best_config["hidden_layer_sizes"],
        alpha=best_config["alpha"],
        max_iter=max_iter,
        random_state=int(tuning_config.seed),
    )
    y_pred_test = final_fit["y_pred_eval"]
    model_final = final_fit["model"]

    mlp_r2 = float(r2_score(y_test_raw, y_pred_test))
    mlp_are = float(average_relative_error(y_test_raw, y_pred_test))

    print(
        "MLP 最优超参数："
        f"hidden_layer_sizes={best_config['hidden_layer_sizes']}, "
        f"alpha={best_config['alpha']}, "
        f"mean val R2="
        f"{tuning_result['candidate_summaries'][tuning_result['best_candidate_idx']]['mean_val_r2']:.6f}"
    )
    print("\n===== MLP 结果 =====")
    print(f"R2  = {mlp_r2:.6f}")
    print(f"ARE = {mlp_are:.6f} %")

    if save_csv_path is not None:
        df_out = pd.DataFrame({
            **test_tracking,
            "true": y_test_raw,
            "MLP_pred": y_pred_test,
        })
        save_dataframe(df_out, save_csv_path)
        print(f"预测文件已保存：{save_csv_path}")

    if save_tuning_records_path is not None:
        save_dataframe(pd.DataFrame(tuning_result["tuning_records"]), save_tuning_records_path)
        print(f"调参审计文件已保存：{save_tuning_records_path}")

    artifact_model_path = None
    artifact_meta_path = None
    artifact_test_inputs_path = None
    artifact_test_targets_path = None
    if save_artifacts:
        if artifact_dir is None:
            raise ValueError("save_artifacts=True 时必须提供 artifact_dir")
        (
            artifact_model_path,
            artifact_meta_path,
            artifact_test_inputs_path,
            artifact_test_targets_path,
        ) = _save_forward_mlp_artifacts(
            model=model_final,
            artifact_dir=artifact_dir,
            data_path=data_path,
            best_config={
                "hidden_layer_sizes": tuple(best_config["hidden_layer_sizes"]),
                "alpha": float(best_config["alpha"]),
            },
            norm_stats=final_fit["norm_stats"],
            idx_tr=idx_tr,
            idx_val=idx_val,
            idx_te=idx_te,
            tuning_config=tuning_config,
            tuning_result=tuning_result,
            resolved_inner_splits=(
                inner_splits if tuning_result["inner_split_strategy"] != "repeated_random" else None
            ),
            X_dev_raw=X_dev_raw,
            y_dev_raw=y_dev_raw,
            X_test_raw=X_test_raw,
            y_test_raw=y_test_raw,
            artifact_extra=artifact_extra,
            artifact_source_files=artifact_source_files,
            model_filename=model_filename,
            meta_filename=meta_filename,
            save_test_slice_flag=save_test_slice,
        )
        print(f"MLP 工件已保存：{artifact_dir}")

    return {
        "r2": mlp_r2,
        "are": mlp_are,
        "best_hidden": tuple(best_config["hidden_layer_sizes"]),
        "best_alpha": float(best_config["alpha"]),
        "best_config": {
            "hidden_layer_sizes": tuple(best_config["hidden_layer_sizes"]),
            "alpha": float(best_config["alpha"]),
        },
        "test_sample_id": test_tracking["sample_id"],
        "test_source_row_number": test_tracking["source_row_number"],
        "y_true": y_test_raw,
        "y_pred": y_pred_test,
        "x_test_raw": X_test_raw,
        "split_indices_payload": split_indices_payload,
        "norm_stats": final_fit["norm_stats"],
        "tuning_protocol": tuning_config_to_dict(tuning_config),
        "trial_budget": int(tuning_config.n_candidates),
        "validation_repeats": int(tuning_result["inner_fold_count"]),
        "selection_metric": tuning_config.selection_metric,
        "tie_break_metric": tuning_config.tie_break_metric,
        "inner_split_strategy": tuning_result["inner_split_strategy"],
        "inner_split_meta": tuning_result["inner_split_meta"],
        "inner_fold_count": int(tuning_result["inner_fold_count"]),
        "tuning_records": tuning_result["tuning_records"],
        "candidate_summaries": tuning_result["candidate_summaries"],
        "artifact_dir": artifact_dir,
        "artifact_model_path": artifact_model_path,
        "artifact_meta_path": artifact_meta_path,
        "artifact_test_inputs_path": artifact_test_inputs_path,
        "artifact_test_targets_path": artifact_test_targets_path,
    }


def main():
    run_dir = create_run_dir("train_mlp")
    output_csv = os.path.join(run_dir, "results_mlp.csv")
    tuning_csv = os.path.join(run_dir, "tuning_records_mlp.csv")
    artifact_dir = os.path.join(run_dir, "artifacts", "forward", "MLP")
    data_path = "data/dataset.xlsx"
    X, y, _ = load_data_with_metadata(data_path)
    split_indices = get_train_val_test_indices(X=X, y=y, random_state=42)
    split_payload = build_single_split_artifact_payload(
        split_indices[0],
        split_indices[1],
        split_indices[2],
        n_samples=len(X),
    )

    write_manifest(
        run_dir,
        script_name="train_mlp.py",
        data_path=data_path,
        seed=42,
        params={
            "hidden_layer_candidates": DEFAULT_HIDDEN_LAYER_CANDIDATES,
            "alpha_candidates": DEFAULT_ALPHA_CANDIDATES,
            "max_iter": 5000,
            "random_state": 42,
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

    res = train_and_eval_mlp(
        data_path=data_path,
        save_csv_path=output_csv,
        split_indices=split_indices,
        save_tuning_records_path=tuning_csv,
        save_artifacts=True,
        artifact_dir=artifact_dir,
        save_test_slice=True,
        artifact_extra={
            "run_dir": run_dir.replace("\\", "/"),
            "reference_output": {
                "path": "results_mlp.csv",
                "prediction_column": "MLP_pred",
                "target_column": "true",
            },
        },
    )

    update_manifest_split_artifact(run_dir, split_artifact_summary=None, split_payload=res["split_indices_payload"])

    outputs = [
        {"path": "results_mlp.csv"},
        {"path": "tuning_records_mlp.csv"},
    ]
    if res["artifact_model_path"] is not None:
        outputs.append({"path": "artifacts/forward/MLP/model.joblib"})
    if res["artifact_meta_path"] is not None:
        outputs.append({"path": "artifacts/forward/MLP/meta.json"})
    if res["artifact_test_inputs_path"] is not None:
        outputs.append({"path": "artifacts/forward/MLP/test_inputs.npy"})
    if res["artifact_test_targets_path"] is not None:
        outputs.append({"path": "artifacts/forward/MLP/test_targets.npy"})

    append_manifest_outputs(run_dir, outputs)
    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()
