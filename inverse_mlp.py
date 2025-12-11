# -*- coding: utf-8 -*-
"""
inverse_mlp.py

反向 MLP 模型（输入+输出 0-1 归一化）：
    输入:  [目标质量 (g/min), 排肥口开度 (mm)]
    输出:  [排肥轴转速 (r/min)]

训练 & 评估方式：
    - 使用 common_utils.load_data() 加载数据
      X_raw: [开度(mm), 转速(r/min)]
      y_raw: [质量(g/min)]
    - 构造“反向问题”:
      输入: [质量, 开度] = X_inv_all
      输出: 转速        = y_inv_all
    - 使用 get_train_val_test_indices 进行 train/val/test 三分
    - 使用 train+val 计算 X, y 的 0–1 归一化参数
    - 内部拟合 y_norm (0~1)，评估前反归一化为真实转速
    - 仅对“实际开度 == 策略开度”的样本评估
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, mean_relative_error


# =========================
# 1. 策略配置（与 KAN/GRNN 保持一致）
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
# 2. 反向 MLP 训练 & 策略一致性评估
# =========================
def train_and_eval_inverse_mlp(
    data_path="data/数据集.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=0,
):
    """
    训练反向 MLP（带输出归一化），并在“策略一致性”样本上评估 R² 和 MRS
    """
    if hidden_layer_candidates is None:
        hidden_layer_candidates = [
            (10,),
            (20,),
            (50,),
            (20, 20),
        ]
    if alpha_candidates is None:
        alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    print("\n=== 训练 反向 MLP (归一化版) ===")

    # --- A. 加载原始数据 ---
    # X_raw: [开度(mm), 转速(r/min)]
    # y_raw: [质量(g/min)]
    X_raw, y_raw = load_data(data_path)

    n_samples = len(y_raw)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(n_samples)

    # 构造“反向问题”的输入输出
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)  # [mass, opening]
    y_inv_all = X_raw[:, 1]                             # speed

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw   = X_inv_all[idx_val]
    X_test_raw  = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw   = y_inv_all[idx_val]
    y_test_raw  = y_inv_all[idx_te]

    # --- B. 使用 train+val 统计归一化参数 ---
    train_full_idx = np.concatenate([idx_tr, idx_val])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    X_min = X_train_full_raw.min(axis=0, keepdims=True)
    X_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = y_train_full_raw.min()
    y_max = y_train_full_raw.max()

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + 1e-8)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + 1e-8)

    # 各子集归一化
    X_train = norm_x(X_train_raw)
    X_val   = norm_x(X_val_raw)
    X_test  = norm_x(X_test_raw)

    y_train_norm = norm_y(y_train_raw)
    y_val_norm   = norm_y(y_val_raw)

    # --- C. 在验证集上搜索 MLP 超参数 ---
    best_r2_val = -np.inf
    best_hidden = None
    best_alpha  = None

    for h in hidden_layer_candidates:
        for a in alpha_candidates:
            mlp = MLPRegressor(
                hidden_layer_sizes=h,
                alpha=a,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )

            # 拟合归一化 y
            mlp.fit(X_train, y_train_norm)

            # 验证集预测 (norm)
            y_val_pred_norm = mlp.predict(X_val)
            # 反归一化回真实转速
            y_val_pred = y_val_pred_norm * (y_max - y_min + 1e-8) + y_min

            r2_val = r2_score(y_val_raw, y_val_pred)

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_hidden = h
                best_alpha  = a

    print(
        f"反向 MLP 最优超参数：hidden_layer_sizes={best_hidden}, "
        f"alpha={best_alpha}, solver='lbfgs', val R²={best_r2_val:.6f}"
    )

    # --- D. 用 train+val 训练最终反向 MLP ---
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

    # --- E. 在 test 集上做“策略一致性”评估 ---
    print("\n>>> 开始验证 (仅筛选符合策略的样本)...")

    # test 集真实物理量
    test_mass       = y_raw[idx_te]
    test_opening    = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    valid_indices = []

    for i in range(len(test_mass)):
        m       = float(test_mass[i])
        real_op = float(test_opening[i])

        strat_op = select_optimal_opening(m)

        if np.isclose(real_op, strat_op, atol=0.1):
            valid_indices.append(i)

    if len(valid_indices) == 0:
        print("测试集中未找到符合当前策略 (20/35/50mm) 的样本。")
        return None

    f_mass      = test_mass[valid_indices]
    f_op        = test_opening[valid_indices]
    f_spd_true  = test_speed_true[valid_indices]

    # 构造输入并归一化
    input_vec  = np.stack([f_mass, f_op], axis=1)
    input_norm = norm_x(input_vec)

    # 预测归一化转速
    pred_spd_norm = mlp_final.predict(input_norm)
    # 反归一化回真实转速
    pred_spd = pred_spd_norm * (y_max - y_min + 1e-8) + y_min

    # 计算指标
    r2  = r2_score(f_spd_true, pred_spd)
    mrs = mean_relative_error(f_spd_true, pred_spd)

    print(f"\n===== 反向 MLP (归一化版) - 策略一致性样本评估 (共 {len(valid_indices)} 个) =====")
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

    return {
        "r2": r2,
        "mrs": mrs,
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,
        "n_valid": len(valid_indices),
        "y_true_valid": f_spd_true,
        "y_pred_valid": pred_spd,
        "mass_valid": f_mass,
    }


if __name__ == "__main__":
    train_and_eval_inverse_mlp()
