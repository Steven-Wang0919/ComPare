# -*- coding: utf-8 -*-
"""
反向 GRNN 模型（带输入+输出 0-1 归一化）：
    输入:  [目标质量 (g/min), 排肥口开度 (mm)]
    输出:  [排肥轴转速 (r/min)]

训练 & 评估方式：
    - 数据划分使用 common_utils.get_train_val_test_indices
    - 归一化范围使用 train+val (train_full_idx)，统一和 KAN
    - 内部拟合的是 y_norm (0~1)，评估时反归一化为真实转速
    - 仅对“实际开度 == 策略开度”的 test 样本进行评估
    - 在 train 上训练, 用 val 集 R² 选择最优 σ

"""

import numpy as np
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, mean_relative_error


# =========================
# 1. 策略配置（与 inverse_kan 保持一致）
# =========================
THRESHOLD_LOW_MID = 2800.0   # 20mm -> 35mm 切换点
THRESHOLD_MID_HIGH = 4800.0  # 35mm -> 50mm 切换点


def select_optimal_opening(target_mass: float) -> float:
    """
    根据目标排肥量，自动决定最佳开度
    """
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    elif target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    else:
        return 50.0


# =========================
# 2. 反向 GRNN 定义
# =========================
class InverseGRNN:
    """
    反向 GRNN:
        X: [目标质量_norm, 开度_norm]
        y: 转速_norm (0~1)
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        X: (n_samples, 2)  -> [mass_norm, opening_norm]
        y: (n_samples,)    -> speed_norm
        """
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def _predict_one(self, x):
        diff = self.X - x
        dist2 = np.sum(diff ** 2, axis=1)
        w = np.exp(-dist2 / (2 * self.sigma ** 2))
        if w.sum() == 0.0:
            # 退化情况：权重全 0 时，返回训练集 y_norm 均值
            return float(self.y.mean())
        return float(np.sum(w * self.y) / w.sum())

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X], dtype=float)


# =========================
# 3. 训练 & 策略一致性评估
# =========================
if __name__ == "__main__":
    np.random.seed(42)

    # --- A. 加载数据 ---
    # X_raw: [开度(mm), 转速(r/min)]
    # y_raw: [质量(g/min)]
    X_raw, y_raw = load_data()
    n_samples = len(y_raw)
    train_idx, val_idx, test_idx = get_train_val_test_indices(n_samples)

    # 构造“反向问题”的输入输出
    # 输入: [质量, 开度]
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)  # (N, 2)
    # 输出: [转速]
    y_inv_all = X_raw[:, 1]                             # (N,)

    # --- B. 使用 train+val 统计归一化参数（与 KAN 对齐） ---
    train_full_idx = np.concatenate([train_idx, val_idx])

    X_train_full_raw = X_inv_all[train_full_idx]  # [mass, opening]
    y_train_full_raw = y_inv_all[train_full_idx]  # speed

    x_min = X_train_full_raw.min(axis=0, keepdims=True)
    x_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = y_train_full_raw.min()
    y_max = y_train_full_raw.max()

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + 1e-8)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + 1e-8)


    # --- C. 在 train/val 上选择最佳 σ（使用 val R²） ---
    sigma_grid = np.linspace(0.1, 4.0, 40)

    # 只用 train 样本拟合, 在 val 上评估 R²
    X_train_raw = X_inv_all[train_idx]
    y_train = y_inv_all[train_idx]

    X_val_raw = X_inv_all[val_idx]
    y_val = y_inv_all[val_idx]

    X_train_norm = norm_x(X_train_raw)
    X_val_norm = norm_x(X_val_raw)

    best_sigma = None
    best_r2_val = -np.inf

    print(">>> 正在为反向 GRNN 选择最优 σ (基于 val R²) ...")
    for s in sigma_grid:
        model_tmp = InverseGRNN(sigma=s)
        model_tmp.fit(X_train_norm, y_train)

        # 在 val 集上评估拟合优度
        y_val_pred = model_tmp.predict(X_val_norm)
        r2_val = r2_score(y_val, y_val_pred)

        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_sigma = s

    print(f"反向 GRNN 最优 σ = {best_sigma:.4f}, val R² = {best_r2_val:.6f}")

    # --- D. 用 train+val 训练最终的反向 GRNN ---

    # --- D. 用 train+val 训练最终的反向 GRNN ---
    X_train_full_norm = norm_x(X_train_full_raw)
    y_train_full_norm = norm_y(y_train_full_raw)

    inv_grnn = InverseGRNN(sigma=best_sigma)
    inv_grnn.fit(X_train_full_norm, y_train_full_norm)

    # --- E. 在测试集上进行“策略一致性”评估 ---
    print("\n>>> 开始验证 (仅筛选符合策略的样本)...")

    test_mass = y_raw[test_idx]        # 真实质量
    test_opening = X_raw[test_idx, 0]  # 实际开度
    test_speed_true = X_raw[test_idx, 1]

    valid_indices = []

    for i in range(len(test_mass)):
        m = float(test_mass[i])
        real_op = float(test_opening[i])

        strat_op = select_optimal_opening(m)

        if np.isclose(real_op, strat_op, atol=0.1):
            valid_indices.append(i)

    if len(valid_indices) == 0:
        print("测试集中未找到符合当前策略 (20/35/50mm) 的样本。")
    else:
        f_mass = test_mass[valid_indices]
        f_op = test_opening[valid_indices]
        f_spd_true = test_speed_true[valid_indices]

        # 组合输入并归一化
        input_vec = np.stack([f_mass, f_op], axis=1)
        input_norm = norm_x(input_vec)

        # 预测归一化转速
        pred_spd_norm = inv_grnn.predict(input_norm)
        # 反归一化得到真实转速
        pred_spd = pred_spd_norm * (y_max - y_min + 1e-8) + y_min

        # 计算指标
        r2 = r2_score(f_spd_true, pred_spd)
        mrs = mean_relative_error(f_spd_true, pred_spd)

        print(f"\n===== 反向 GRNN (归一化版) - 策略一致性样本评估 (共 {len(valid_indices)} 个) =====")
        print(f"R² Score: {r2:.4f}")
        print(f"平均相对误差 (MRS): {mrs:.4f}%")

        print("\n[典型样本详情]")
        print(f"{'目标质量':<10} | {'策略开度':<10} | {'实际开度':<10} | "
              f"{'预测转速':<10} | {'实际转速':<10} | {'误差(%)'}")
        for k in range(min(5, len(f_mass))):
            err = abs(pred_spd[k] - f_spd_true[k]) / (f_spd_true[k] + 1e-8) * 100
            print(
                f"{f_mass[k]:<10.1f} | "
                f"{f_op[k]:<10.1f} | "
                f"{f_op[k]:<10.1f} | "
                f"{pred_spd[k]:<10.2f} | "
                f"{f_spd_true[k]:<10.2f} | "
                f"{err:.2f}%"
            )
