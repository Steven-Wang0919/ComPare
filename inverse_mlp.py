# -*- coding: utf-8 -*-
"""
inverse_mlp.py

反向 MLP 模型（严格披露版）：
    输入:  [目标质量 (g/min), 排肥口开度 (mm)]
    输出:  [排肥轴转速 (r/min)]

本版修订重点：
1. 保留“主结果 = 策略一致子集；补充结果 = 全测试集”的评估框架
2. 显式报告：
   - 主结果样本数 n_main
   - 全测试集样本数 n_all
   - 主结果占比 main_ratio
   - 主结果与全测试集的 R² / ARE 并列输出
3. 增加策略一致子集在不同开度上的样本分布，避免选择性报告风险
4. 返回更完整的统计信息，便于 compare / 论文表格 / 附录使用
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, average_relative_error


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
    stats["other"] = int(
        len(openings)
        - sum(stats[k] for k in ["20mm", "35mm", "50mm"])
    )
    return stats


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

    print("\n=== 训练 反向 MLP（严格披露版，转速优先口径） ===")

    X_raw, y_raw = load_data(data_path)

    n_samples = len(y_raw)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(n_samples)

    # 反向任务：
    # 输入 = [目标质量, 开度]
    # 输出 = 转速
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]
    y_test_raw = y_inv_all[idx_te]

    # 与原实现一致：反向任务的归一化统计量使用 train+val
    # 若后续希望进一步提高协议严格性，可单独再改成仅 train
    train_full_idx = np.concatenate([idx_tr, idx_val])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    X_min = X_train_full_raw.min(axis=0, keepdims=True)
    X_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_full_raw.min())
    y_max = float(y_train_full_raw.max())

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + EPS)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + EPS)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + EPS) + y_min

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    y_train_norm = norm_y(y_train_raw)

    best_r2_val = -np.inf
    best_hidden = None
    best_alpha = None

    print(">>> 开始搜索反向 MLP 最优超参数（基于 val R²，原始物理量空间）...")
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

    # 全测试集上的真实物理量
    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    # 全测试集预测
    X_test_norm = norm_x(X_test_raw)
    pred_test_norm = mlp_final.predict(X_test_norm)
    pred_test_speed = denorm_y(pred_test_norm)

    # 全测试集指标
    r2_all = _safe_r2(test_speed_true, pred_test_speed)
    are_all = _safe_are(test_speed_true, pred_test_speed)
    n_all = int(len(test_speed_true))

    # 策略一致子集
    strategy_opening_all = np.array(
        [select_optimal_opening(float(m)) for m in test_mass],
        dtype=float
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

    # 分布披露：全测试集 / 主结果子集的开度分布
    opening_dist_all = _count_openings(test_opening)
    opening_dist_main = _count_openings(opening_main)

    print("\n===== 反向 MLP（转速优先口径）测试结果 =====")
    if n_main > 0:
        print(
            f"主结果（策略一致子集）: n = {n_main:3d} / {n_all:3d} "
            f"({main_ratio * 100:.2f}%), R² = {r2_main:.4f}, ARE = {are_main:.4f}%"
        )
    else:
        print(
            f"主结果（策略一致子集）: n =   0 / {n_all:3d} "
            f"(0.00%), R² = NaN, ARE = NaN"
        )

    print(
        f"补充结果（全测试集）  : n = {n_all:3d}, "
        f"R² = {r2_all:.4f}, ARE = {are_all:.4f}%"
    )

    print("\n--- 开度分布披露 ---")
    print(
        "全测试集开度分布: "
        f"20mm={opening_dist_all['20mm']}, "
        f"35mm={opening_dist_all['35mm']}, "
        f"50mm={opening_dist_all['50mm']}, "
        f"other={opening_dist_all['other']}"
    )
    print(
        "主结果子集开度分布: "
        f"20mm={opening_dist_main['20mm']}, "
        f"35mm={opening_dist_main['35mm']}, "
        f"50mm={opening_dist_main['50mm']}, "
        f"other={opening_dist_main['other']}"
    )

    print("\n--- 结果解释建议 ---")
    print("主结果基于‘实际开度 = 策略推荐开度’的测试样本。")
    print("为避免选择性报告，应始终与全测试集结果并列呈现，并说明主结果样本占比。")

    return {
        # 主结果
        "r2_main": float(r2_main) if not np.isnan(r2_main) else np.nan,
        "are_main": float(are_main) if not np.isnan(are_main) else np.nan,
        "n_main": n_main,
        "main_ratio": float(main_ratio) if not np.isnan(main_ratio) else np.nan,

        # 全测试集
        "r2_all": float(r2_all) if not np.isnan(r2_all) else np.nan,
        "are_all": float(are_all) if not np.isnan(are_all) else np.nan,
        "n_all": n_all,

        # 超参数
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,

        # 全测试集明细
        "y_true_all": np.asarray(test_speed_true),
        "y_pred_all": np.asarray(pred_test_speed),
        "mass_all": np.asarray(test_mass),
        "opening_all": np.asarray(test_opening),
        "strategy_opening_all": np.asarray(strategy_opening_all),

        # 主结果子集明细
        "y_true_main": np.asarray(y_true_main),
        "y_pred_main": np.asarray(y_pred_main),
        "mass_main": np.asarray(mass_main),
        "opening_main": np.asarray(opening_main),
        "strategy_opening_main": np.asarray(strategy_opening_main),

        # 策略一致掩码
        "policy_mask": np.asarray(policy_mask),

        # 披露统计
        "opening_dist_all": opening_dist_all,
        "opening_dist_main": opening_dist_main,
    }


if __name__ == "__main__":
    train_and_eval_inverse_mlp()