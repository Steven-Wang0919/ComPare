# -*- coding: utf-8 -*-
"""
inverse_mlp.py

反向 MLP 模型（输入+输出 0-1 归一化）：
    输入:  [目标质量 (g/min), 排肥口开度 (mm)]
    输出:  [排肥轴转速 (r/min)]
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, average_relative_error


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0


def select_optimal_opening(target_mass: float) -> float:
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    elif target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    else:
        return 50.0


def train_and_eval_inverse_mlp(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=0,
):
    if hidden_layer_candidates is None:
        hidden_layer_candidates = [
            (10,),
            (20,),
            (50,),
            (20, 20),
        ]
    if alpha_candidates is None:
        alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    print("\n=== 训练 反向 MLP（归一化版，转速优先口径） ===")

    X_raw, y_raw = load_data(data_path)

    n_samples = len(y_raw)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(n_samples)

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]
    y_test_raw = y_inv_all[idx_te]

    train_full_idx = np.concatenate([idx_tr, idx_val])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    X_min = X_train_full_raw.min(axis=0, keepdims=True)
    X_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_full_raw.min())
    y_max = float(y_train_full_raw.max())

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + 1e-8)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + 1e-8)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + 1e-8) + y_min

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    y_train_norm = norm_y(y_train_raw)

    best_r2_val = -np.inf
    best_hidden = None
    best_alpha = None

    print(">>> 开始搜索反向 MLP 最优超参数（基于 val R²）...")
    for h in hidden_layer_candidates:
        for a in alpha_candidates:
            mlp = MLPRegressor(
                hidden_layer_sizes=h,
                alpha=a,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )

            mlp.fit(X_train, y_train_norm)

            y_val_pred_norm = mlp.predict(X_val)
            y_val_pred = denorm_y(y_val_pred_norm)
            r2_val = r2_score(y_val_raw, y_val_pred)

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_hidden = h
                best_alpha = a

    if best_hidden is None or best_alpha is None:
        raise RuntimeError("反向 MLP 超参数搜索失败：best_hidden / best_alpha 为空")

    print(
        f"反向 MLP 最优超参数：hidden_layer_sizes={best_hidden}, "
        f"alpha={best_alpha}, solver='lbfgs', val R²={best_r2_val:.6f}"
    )

    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])

    X_train_val = norm_x(X_train_val_raw)
    y_train_val_norm = norm_y(y_train_val_raw)

    mlp_final = MLPRegressor(
        hidden_layer_sizes=best_hidden,
        alpha=best_alpha,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp_final.fit(X_train_val, y_train_val_norm)

    print("\n>>> 开始测试集评估（主结果=策略一致子集，补充=全测试集）...")

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    X_test_norm = norm_x(X_test_raw)
    pred_test_norm = mlp_final.predict(X_test_norm)
    pred_test_speed = denorm_y(pred_test_norm)

    r2_all = r2_score(test_speed_true, pred_test_speed)
    are_all = average_relative_error(test_speed_true, pred_test_speed)

    strategy_opening_all = np.array(
        [select_optimal_opening(float(m)) for m in test_mass],
        dtype=float
    )
    policy_mask = np.isclose(test_opening, strategy_opening_all, atol=0.1)

    n_main = int(policy_mask.sum())

    if n_main > 0:
        mass_main = test_mass[policy_mask]
        opening_main = test_opening[policy_mask]
        strategy_opening_main = strategy_opening_all[policy_mask]
        y_true_main = test_speed_true[policy_mask]
        y_pred_main = pred_test_speed[policy_mask]

        r2_main = r2_score(y_true_main, y_pred_main)
        are_main = average_relative_error(y_true_main, y_pred_main)
    else:
        mass_main = np.array([], dtype=float)
        opening_main = np.array([], dtype=float)
        strategy_opening_main = np.array([], dtype=float)
        y_true_main = np.array([], dtype=float)
        y_pred_main = np.array([], dtype=float)
        r2_main = np.nan
        are_main = np.nan

    print("\n===== 反向 MLP（转速优先口径）测试结果 =====")
    if n_main > 0:
        print(f"主结果（策略一致子集）: n = {n_main:3d}, R² = {r2_main:.4f}, ARE = {are_main:.4f}%")
    else:
        print("主结果（策略一致子集）: n =   0, R² = NaN, ARE = NaN")

    print(f"补充结果（全测试集）  : n = {len(test_speed_true):3d}, R² = {r2_all:.4f}, ARE = {are_all:.4f}%")

    return {
        "r2_main": float(r2_main) if not np.isnan(r2_main) else np.nan,
        "are_main": float(are_main) if not np.isnan(are_main) else np.nan,
        "n_main": n_main,

        "r2_all": float(r2_all),
        "are_all": float(are_all),
        "n_all": int(len(test_speed_true)),

        "best_hidden": best_hidden,
        "best_alpha": best_alpha,

        "y_true_all": np.asarray(test_speed_true),
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
    }


if __name__ == "__main__":
    train_and_eval_inverse_mlp()