# -*- coding: utf-8 -*-
"""
inverse_mlp.py

Inverse MLP with a shared fair tuning protocol.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from common_utils import (
    average_relative_error,
    combine_train_val_indices,
    get_train_val_test_indices,
    load_data,
    validate_predefined_split_indices,
)
from fair_tuning import (
    build_inner_repeated_splits,
    ensure_fair_tuning_config,
    infer_inner_val_ratio,
    run_fair_tuning,
    tuning_config_to_dict,
)
from run_utils import append_manifest_outputs, create_run_dir, save_dataframe, write_manifest


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0
EPS = 1e-8
DEFAULT_HIDDEN_LAYER_CANDIDATES = [(8,), (16,), (32,), (16, 16), (32, 16), (32, 32)]
DEFAULT_ALPHA_CANDIDATES = [1e-6, 1e-5, 1e-4, 1e-3]


def select_optimal_opening(target_mass: float) -> float:
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    if target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    return 50.0


def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]):
        return np.nan
    return float(r2_score(y_true, y_pred))


def _safe_are(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(average_relative_error(y_true, y_pred))


def _count_openings(openings, opening_values=(20.0, 35.0, 50.0), atol=0.1):
    openings = np.asarray(openings, dtype=float)
    stats = {}
    for v in opening_values:
        stats[f"{int(v)}mm"] = int(np.isclose(openings, v, atol=atol).sum())
    stats["other"] = int(len(openings) - sum(stats[k] for k in ["20mm", "35mm", "50mm"]))
    return stats


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


def _fit_predict_inverse_mlp(
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
    y_min = float(y_train_raw.min())
    y_max = float(y_train_raw.max())

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
        "y_pred_eval": y_pred_eval,
        "norm_stats": {
            "X_min": X_min,
            "X_max": X_max,
            "y_min": y_min,
            "y_max": y_max,
        },
    }


def train_and_eval_inverse_mlp(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=42,
    save_outputs_dir=None,
    split_indices=None,
    tuning_config=None,
    save_tuning_records_path=None,
):
    print("\n=== 训练 反向 MLP（公平调参协议） ===")

    X_raw, y_raw = load_data(data_path)
    if split_indices is None:
        idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X_raw, y=y_raw, random_state=random_state)
    else:
        idx_tr, idx_val, idx_te = validate_predefined_split_indices(
            len(X_raw), split_indices[0], split_indices[1], split_indices[2]
        )

    inner_val_ratio = infer_inner_val_ratio(idx_tr, idx_val)
    tuning_config = ensure_fair_tuning_config(
        tuning_config,
        seed=random_state,
        inner_val_ratio=inner_val_ratio,
    )

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    dev_idx = combine_train_val_indices(idx_tr, idx_val)
    X_dev_raw = X_inv_all[dev_idx]
    y_dev_raw = y_inv_all[dev_idx]
    X_test_raw = X_inv_all[idx_te]
    y_test_speed_true = y_inv_all[idx_te]

    inner_splits = build_inner_repeated_splits(X_dev_raw, y_dev_raw, tuning_config)
    candidate_configs = _build_candidate_configs(hidden_layer_candidates, alpha_candidates)

    def eval_candidate_fn(*, config, idx_train, idx_val, repeat_idx):
        res = _fit_predict_inverse_mlp(
            X_dev_raw[idx_train],
            y_dev_raw[idx_train],
            X_dev_raw[idx_val],
            hidden_layer_sizes=config["hidden_layer_sizes"],
            alpha=config["alpha"],
            max_iter=max_iter,
            random_state=int(tuning_config.seed) + repeat_idx,
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
        model_name="inverse_MLP",
        task_name="inverse",
    )
    best_config = tuning_result["best_config"]

    final_fit = _fit_predict_inverse_mlp(
        X_dev_raw,
        y_dev_raw,
        X_test_raw,
        hidden_layer_sizes=best_config["hidden_layer_sizes"],
        alpha=best_config["alpha"],
        max_iter=max_iter,
        random_state=int(tuning_config.seed),
    )
    pred_test_speed = final_fit["y_pred_eval"]

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]

    r2_all = _safe_r2(y_test_speed_true, pred_test_speed)
    are_all = _safe_are(y_test_speed_true, pred_test_speed)
    n_all = int(len(y_test_speed_true))

    strategy_opening_all = np.array(
        [select_optimal_opening(float(m)) for m in test_mass],
        dtype=float,
    )
    policy_mask = np.isclose(test_opening, strategy_opening_all, atol=0.1)

    n_main = int(policy_mask.sum())
    main_ratio = float(n_main / n_all) if n_all > 0 else np.nan

    if n_main > 0:
        mass_main = test_mass[policy_mask]
        opening_main = test_opening[policy_mask]
        strategy_opening_main = strategy_opening_all[policy_mask]
        y_true_main = y_test_speed_true[policy_mask]
        y_pred_main = pred_test_speed[policy_mask]
        r2_main = _safe_r2(y_true_main, y_pred_main)
        are_main = _safe_are(y_true_main, y_pred_main)
    else:
        mass_main = np.array([], dtype=float)
        opening_main = np.array([], dtype=float)
        strategy_opening_main = np.array([], dtype=float)
        y_true_main = np.array([], dtype=float)
        y_pred_main = np.array([], dtype=float)
        r2_main = np.nan
        are_main = np.nan

    opening_dist_all = _count_openings(test_opening)
    opening_dist_main = _count_openings(opening_main)

    print(
        "反向 MLP 最优超参数："
        f"hidden_layer_sizes={best_config['hidden_layer_sizes']}, "
        f"alpha={best_config['alpha']}, "
        f"mean val R2={tuning_result['candidate_summaries'][tuning_result['best_candidate_idx']]['mean_val_r2']:.6f}"
    )

    if save_outputs_dir is not None:
        df_all = pd.DataFrame({
            "target_mass_g_min": test_mass,
            "actual_opening_mm": test_opening,
            "strategy_opening_mm": strategy_opening_all,
            "true_speed_r_min": y_test_speed_true,
            "inverse_MLP_pred": pred_test_speed,
            "policy_match": policy_mask.astype(int),
        })
        save_dataframe(df_all, os.path.join(save_outputs_dir, "inverse_mlp_predictions_all.csv"))

        df_main = pd.DataFrame({
            "target_mass_g_min": mass_main,
            "actual_opening_mm": opening_main,
            "strategy_opening_mm": strategy_opening_main,
            "true_speed_r_min": y_true_main,
            "inverse_MLP_pred": y_pred_main,
        })
        save_dataframe(df_main, os.path.join(save_outputs_dir, "inverse_mlp_predictions_main.csv"))

    if save_tuning_records_path is not None:
        save_dataframe(pd.DataFrame(tuning_result["tuning_records"]), save_tuning_records_path)

    return {
        "r2_main": r2_main,
        "are_main": are_main,
        "n_main": n_main,
        "main_ratio": main_ratio,
        "r2_all": r2_all,
        "are_all": are_all,
        "n_all": n_all,
        "best_hidden": tuple(best_config["hidden_layer_sizes"]),
        "best_alpha": float(best_config["alpha"]),
        "best_config": {
            "hidden_layer_sizes": tuple(best_config["hidden_layer_sizes"]),
            "alpha": float(best_config["alpha"]),
        },
        "y_true_all": np.asarray(y_test_speed_true),
        "y_pred_all": np.asarray(pred_test_speed),
        "mass_all": np.asarray(test_mass),
        "opening_all": np.asarray(test_opening),
        "strategy_opening_all": np.asarray(strategy_opening_all),
        "y_true_main": np.asarray(y_true_main),
        "y_pred_main": np.asarray(y_pred_main),
        "mass_main": np.asarray(mass_main),
        "opening_main": np.asarray(opening_main),
        "strategy_opening_main": np.asarray(strategy_opening_main),
        "policy_mask": np.asarray(policy_mask),
        "opening_dist_all": opening_dist_all,
        "opening_dist_main": opening_dist_main,
        "tuning_protocol": tuning_config_to_dict(tuning_config),
        "trial_budget": int(tuning_config.n_candidates),
        "validation_repeats": int(tuning_config.n_repeats),
        "selection_metric": tuning_config.selection_metric,
        "tie_break_metric": tuning_config.tie_break_metric,
        "tuning_records": tuning_result["tuning_records"],
        "candidate_summaries": tuning_result["candidate_summaries"],
        "norm_stats": final_fit["norm_stats"],
    }


def main():
    run_dir = create_run_dir("inverse_mlp")
    tuning_csv = os.path.join(run_dir, "tuning_records_inverse_mlp.csv")

    write_manifest(
        run_dir,
        script_name="inverse_mlp.py",
        data_path="data/dataset.xlsx",
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
    )

    train_and_eval_inverse_mlp(
        save_outputs_dir=run_dir,
        save_tuning_records_path=tuning_csv,
    )

    append_manifest_outputs(
        run_dir,
        [
            {"path": "inverse_mlp_predictions_all.csv"},
            {"path": "inverse_mlp_predictions_main.csv"},
            {"path": "tuning_records_inverse_mlp.csv"},
        ],
    )

    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()
