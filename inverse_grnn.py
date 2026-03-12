# -*- coding: utf-8 -*-
"""
inverse_grnn.py

反向 GRNN：
- 主结果 = 策略一致子集
- 补充结果 = 全测试集
- 独立运行时输出到 runs/<timestamp>_inverse_grnn/
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, average_relative_error
from run_utils import append_manifest_outputs, create_run_dir, save_dataframe, write_manifest


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0
EPS = 1e-8


def select_optimal_opening(target_mass: float) -> float:
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    elif target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    else:
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


class InverseGRNN:
    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        if self.X.ndim != 2:
            raise ValueError("X 必须是二维数组")
        if len(self.X) != len(self.y):
            raise ValueError("X 和 y 的样本数必须一致")
        if len(self.X) == 0:
            raise ValueError("训练数据不能为空")

    def _predict_one(self, x):
        diff = self.X - x
        dist2 = np.sum(diff ** 2, axis=1)
        w = np.exp(-dist2 / (2.0 * self.sigma ** 2))
        w_sum = w.sum()
        if w_sum <= EPS:
            return float(self.y.mean())
        return float(np.sum(w * self.y) / w_sum)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x) for x in X], dtype=float)


def train_and_eval_inverse_grnn(
    data_path="data/dataset.xlsx",
    sigma_grid=None,
    save_outputs_dir=None,
):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.1, 4.0, 40)

    sigma_grid = np.asarray(sigma_grid, dtype=float).reshape(-1)
    if len(sigma_grid) == 0:
        raise ValueError("sigma_grid 不能为空")

    print("\n=== 训练 反向 GRNN（train-only normalization，转速优先口径） ===")

    X_raw, y_raw = load_data(data_path)
    train_idx, val_idx, test_idx = get_train_val_test_indices(X=X_raw, y=y_raw)

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[train_idx]
    y_train_raw = y_inv_all[train_idx]
    X_val_raw = X_inv_all[val_idx]
    y_val_raw = y_inv_all[val_idx]

    x_min = X_train_raw.min(axis=0, keepdims=True)
    x_max = X_train_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_raw.min())
    y_max = float(y_train_raw.max())

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + EPS)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + EPS)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + EPS) + y_min

    X_train_norm = norm_x(X_train_raw)
    X_val_norm = norm_x(X_val_raw)
    y_train_norm = norm_y(y_train_raw)

    best_sigma = None
    best_r2_val = -np.inf

    for s in sigma_grid:
        model_tmp = InverseGRNN(sigma=float(s))
        model_tmp.fit(X_train_norm, y_train_norm)
        y_val_pred_norm = model_tmp.predict(X_val_norm)
        y_val_pred_raw = denorm_y(y_val_pred_norm)
        r2_val = r2_score(y_val_raw, y_val_pred_raw)

        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_sigma = float(s)

    if best_sigma is None:
        raise RuntimeError("反向 GRNN 超参数搜索失败：best_sigma 为空")

    print(f"反向 GRNN 最优 σ = {best_sigma:.4f}, val R² = {best_r2_val:.6f}")

    train_full_idx = np.concatenate([train_idx, val_idx])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    X_train_full_norm = norm_x(X_train_full_raw)
    y_train_full_norm = norm_y(y_train_full_raw)

    inv_grnn = InverseGRNN(sigma=best_sigma)
    inv_grnn.fit(X_train_full_norm, y_train_full_norm)

    test_mass = y_raw[test_idx]
    test_opening = X_raw[test_idx, 0]
    test_speed_true = X_raw[test_idx, 1]

    X_test_raw = np.stack([test_mass, test_opening], axis=1)
    X_test_norm = norm_x(X_test_raw)

    pred_test_norm = inv_grnn.predict(X_test_norm)
    pred_test_speed = denorm_y(pred_test_norm)

    r2_all = _safe_r2(test_speed_true, pred_test_speed)
    are_all = _safe_are(test_speed_true, pred_test_speed)
    n_all = int(len(test_speed_true))

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
        y_true_main = test_speed_true[policy_mask]
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

    if save_outputs_dir is not None:
        df_all = pd.DataFrame({
            "target_mass_g_min": test_mass,
            "actual_opening_mm": test_opening,
            "strategy_opening_mm": strategy_opening_all,
            "true_speed_r_min": test_speed_true,
            "inverse_GRNN_pred": pred_test_speed,
            "policy_match": policy_mask.astype(int),
        })
        save_dataframe(df_all, os.path.join(save_outputs_dir, "inverse_grnn_predictions_all.csv"))

        df_main = pd.DataFrame({
            "target_mass_g_min": mass_main,
            "actual_opening_mm": opening_main,
            "strategy_opening_mm": strategy_opening_main,
            "true_speed_r_min": y_true_main,
            "inverse_GRNN_pred": y_pred_main,
        })
        save_dataframe(df_main, os.path.join(save_outputs_dir, "inverse_grnn_predictions_main.csv"))

    return {
        "r2_main": r2_main,
        "are_main": are_main,
        "n_main": n_main,
        "main_ratio": main_ratio,
        "r2_all": r2_all,
        "are_all": are_all,
        "n_all": n_all,
        "best_sigma": best_sigma,
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
        "opening_dist_all": opening_dist_all,
        "opening_dist_main": opening_dist_main,
    }


def main():
    run_dir = create_run_dir("inverse_grnn")

    write_manifest(
        run_dir,
        script_name="inverse_grnn.py",
        data_path="data/dataset.xlsx",
        seed=None,
        params={"sigma_grid": "np.linspace(0.1, 4.0, 40)"},
    )

    train_and_eval_inverse_grnn(save_outputs_dir=run_dir)

    append_manifest_outputs(
        run_dir,
        [
            {"path": "inverse_grnn_predictions_all.csv"},
            {"path": "inverse_grnn_predictions_main.csv"},
        ],
    )

    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()