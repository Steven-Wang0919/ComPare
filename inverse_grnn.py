# -*- coding: utf-8 -*-
"""
inverse_grnn.py

反向 GRNN 模型（输入+输出 0-1 归一化）：
    输入:  [目标质量 (g/min), 排肥口开度 (mm)]
    输出:  [排肥轴转速 (r/min)]

任务定义（与论文口径一致）：
    本模型服务于“转速优先”控制方法：
        1) 先根据目标质量确定策略开度
        2) 在策略开度确定后，再调节排肥轴转速

因此：
    - 主评估对象：满足“实际开度 == 策略开度”的测试样本（策略一致子集）
    - 补充评估对象：全测试集（仅作透明展示，不作为主结论）

训练 & 评估方式：
    - 使用 common_utils.load_data() 加载数据
      X_raw: [开度(mm), 转速(r/min)]
      y_raw: [质量(g/min)]
    - 构造“反向问题”:
      输入: [质量, 开度] = X_inv_all
      输出: 转速        = y_inv_all
    - 使用 get_train_val_test_indices 进行 train/val/test 三分
    - 使用 train+val 计算 X, y 的 0–1 归一化参数
    - 模型内部拟合 y_norm (0~1)，评估前反归一化为真实转速
    - 在 train 上训练，用 val 集 R² 选择最优 σ
"""

import numpy as np
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


class InverseGRNN:
    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)

        if self.X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状应为 (n_samples, n_features)")
        if self.y.ndim != 1:
            raise ValueError("y 必须是一维数组，形状应为 (n_samples,)")
        if len(self.X) != len(self.y):
            raise ValueError("X 和 y 的样本数必须一致")
        if len(self.X) == 0:
            raise ValueError("训练数据不能为空")

    def _predict_one(self, x):
        diff = self.X - x
        dist2 = np.sum(diff ** 2, axis=1)
        w = np.exp(-dist2 / (2.0 * self.sigma ** 2))

        if w.sum() == 0.0:
            return float(self.y.mean())

        return float(np.sum(w * self.y) / w.sum())

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x) for x in X], dtype=float)


def train_and_eval_inverse_grnn(
    data_path="data/dataset.xlsx",
    sigma_grid=None,
):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.1, 4.0, 40)

    sigma_grid = np.asarray(sigma_grid, dtype=float).reshape(-1)
    if len(sigma_grid) == 0:
        raise ValueError("sigma_grid 不能为空")

    print("\n=== 训练 反向 GRNN（归一化版，转速优先口径） ===")

    X_raw, y_raw = load_data(data_path)

    n_samples = len(y_raw)
    train_idx, val_idx, test_idx = get_train_val_test_indices(n_samples)

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    train_full_idx = np.concatenate([train_idx, val_idx])

    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    x_min = X_train_full_raw.min(axis=0, keepdims=True)
    x_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_full_raw.min())
    y_max = float(y_train_full_raw.max())

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + 1e-8)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + 1e-8)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + 1e-8) + y_min

    X_train_raw = X_inv_all[train_idx]
    y_train_raw = y_inv_all[train_idx]

    X_val_raw = X_inv_all[val_idx]
    y_val_raw = y_inv_all[val_idx]

    X_train_norm = norm_x(X_train_raw)
    X_val_norm = norm_x(X_val_raw)
    y_train_norm = norm_y(y_train_raw)

    best_sigma = None
    best_r2_val = -np.inf

    print(">>> 正在为反向 GRNN 选择最优 σ（基于 val R²）...")
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

    X_train_full_norm = norm_x(X_train_full_raw)
    y_train_full_norm = norm_y(y_train_full_raw)

    inv_grnn = InverseGRNN(sigma=best_sigma)
    inv_grnn.fit(X_train_full_norm, y_train_full_norm)

    print("\n>>> 开始测试集评估（主结果=策略一致子集，补充=全测试集）...")

    test_mass = y_raw[test_idx]
    test_opening = X_raw[test_idx, 0]
    test_speed_true = X_raw[test_idx, 1]

    X_test_raw = np.stack([test_mass, test_opening], axis=1)
    X_test_norm = norm_x(X_test_raw)

    pred_test_norm = inv_grnn.predict(X_test_norm)
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

    print("\n===== 反向 GRNN（转速优先口径）测试结果 =====")
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

        "best_sigma": float(best_sigma),

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
    train_and_eval_inverse_grnn()