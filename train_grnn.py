# -*- coding: utf-8 -*-
"""
train_grnn.py

Forward GRNN with a shared fair tuning protocol.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from common_utils import (
    average_relative_error,
    combine_train_val_indices,
    get_train_val_test_indices,
    load_data,
    validate_predefined_split_indices,
)
from fair_tuning import (
    ensure_fair_tuning_config,
    infer_inner_val_ratio,
    prepare_inner_cv,
    run_fair_tuning,
    tuning_config_to_dict,
)
from run_utils import append_manifest_outputs, create_run_dir, save_dataframe, write_manifest


EPS = 1e-8
DEFAULT_SIGMA_GRID = np.linspace(0.10, 4.00, 24)


class GRNN:
    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def _predict_one(self, x):
        diff = self.X - x
        dist2 = np.sum(diff ** 2, axis=1)
        w = np.exp(-dist2 / (2 * self.sigma ** 2))
        w_sum = w.sum()
        if w_sum <= EPS:
            return float(np.mean(self.y))
        return float(np.sum(w * self.y) / w_sum)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X], dtype=float)


def _build_candidate_configs(sigma_grid=None):
    sigma_grid = np.asarray(
        DEFAULT_SIGMA_GRID if sigma_grid is None else sigma_grid,
        dtype=float,
    ).reshape(-1)
    return [{"sigma": float(s)} for s in sigma_grid]


def _fit_predict_forward_grnn(X_train_raw, y_train_raw, X_eval_raw, *, sigma):
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

    model = GRNN(sigma=float(sigma))
    model.fit(norm_x(X_train_raw), norm_y(y_train_raw))
    y_pred_eval = denorm_y(model.predict(norm_x(X_eval_raw)))

    return {
        "y_pred_eval": y_pred_eval,
        "norm_stats": {
            "X_min": X_min,
            "X_max": X_max,
            "y_min": y_min,
            "y_max": y_max,
        },
    }


def train_and_eval_grnn(
    data_path="data/dataset.xlsx",
    sigma_grid=None,
    save_csv_path=None,
    split_indices=None,
    tuning_config=None,
    save_tuning_records_path=None,
    random_state=42,
    inner_splits=None,
    inner_split_strategy=None,
    inner_split_meta=None,
):
    print("\n=== 训练 GRNN（公平调参协议） ===")

    X, y = load_data(data_path)
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

    inner_splits, inner_split_strategy, inner_split_meta = prepare_inner_cv(
        X_dev_raw,
        y_dev_raw,
        tuning_config,
        inner_splits=inner_splits,
        inner_split_strategy=inner_split_strategy,
        inner_split_meta=inner_split_meta,
    )
    candidate_configs = _build_candidate_configs(sigma_grid)

    def eval_candidate_fn(*, config, idx_train, idx_val, fold_id, split_meta):
        res = _fit_predict_forward_grnn(
            X_dev_raw[idx_train],
            y_dev_raw[idx_train],
            X_dev_raw[idx_val],
            sigma=config["sigma"],
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
        model_name="GRNN",
        task_name="forward",
        inner_split_strategy=inner_split_strategy,
        inner_split_meta=inner_split_meta,
    )
    best_config = tuning_result["best_config"]

    final_fit = _fit_predict_forward_grnn(
        X_dev_raw,
        y_dev_raw,
        X_test_raw,
        sigma=best_config["sigma"],
    )
    y_pred_test = final_fit["y_pred_eval"]

    grnn_r2 = float(r2_score(y_test_raw, y_pred_test))
    grnn_are = float(average_relative_error(y_test_raw, y_pred_test))

    print(
        f"GRNN 最优 sigma = {best_config['sigma']:.6f}, "
        f"mean val R2 = {tuning_result['candidate_summaries'][tuning_result['best_candidate_idx']]['mean_val_r2']:.6f}"
    )
    print("\n===== GRNN 结果 =====")
    print(f"R2  = {grnn_r2:.6f}")
    print(f"ARE = {grnn_are:.6f} %")

    if save_csv_path is not None:
        df_out = pd.DataFrame({
            "true": y_test_raw,
            "GRNN_pred": y_pred_test,
        })
        save_dataframe(df_out, save_csv_path)
        print(f"预测文件已保存：{save_csv_path}")

    if save_tuning_records_path is not None:
        save_dataframe(pd.DataFrame(tuning_result["tuning_records"]), save_tuning_records_path)
        print(f"调参审计文件已保存：{save_tuning_records_path}")

    return {
        "r2": grnn_r2,
        "are": grnn_are,
        "best_sigma": float(best_config["sigma"]),
        "best_config": {"sigma": float(best_config["sigma"])},
        "y_true": y_test_raw,
        "y_pred": y_pred_test,
        "x_test_raw": X_test_raw,
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
    }


def main():
    run_dir = create_run_dir("train_grnn")
    output_csv = os.path.join(run_dir, "results_grnn.csv")
    tuning_csv = os.path.join(run_dir, "tuning_records_grnn.csv")

    write_manifest(
        run_dir,
        script_name="train_grnn.py",
        data_path="data/dataset.xlsx",
        seed=42,
        params={
            "sigma_grid": [float(x) for x in DEFAULT_SIGMA_GRID.tolist()],
            "fair_tuning": {
                "n_candidates": 24,
                "n_repeats": 5,
                "selection_metric": "mean_val_r2",
                "tie_break_metric": "mean_val_are",
                "budget_profile": "high",
            },
        },
    )

    train_and_eval_grnn(
        save_csv_path=output_csv,
        save_tuning_records_path=tuning_csv,
    )

    append_manifest_outputs(
        run_dir,
        [
            {"path": "results_grnn.csv"},
            {"path": "tuning_records_grnn.csv"},
        ],
    )

    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()
