# -*- coding: utf-8 -*-
"""
训练 & 评估 GRNN（统一口径版）

修订说明：
- 使用 train/val/test 三分数据
- X 使用 train 统计量做 0–1 归一化
- y 也使用 train 统计量做 0–1 归一化（与 KAN / 修订后的 MLP 一致）
- 在验证集上选择最优 sigma
- 最终测试指标统一在“原始物理量空间”上计算：ARE 和 R²
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, average_relative_error


EPS = 1e-8


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


def train_and_eval_grnn(
    data_path="data/dataset.xlsx",
    sigma_grid=None,
    save_csv_path="results_grnn.csv",
):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.1, 4.0, 40)

    print("\n=== 训练 GRNN（统一 y 归一化口径） ===")

    X, y = load_data(data_path)

    idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X, y=y)
    X_train_raw, y_train_raw = X[idx_tr], y[idx_tr]
    X_val_raw, y_val_raw = X[idx_val], y[idx_val]
    X_test_raw, y_test_raw = X[idx_te], y[idx_te]

    # -----------------------------
    # 只使用 train 统计量做归一化
    # -----------------------------
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

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    X_test = norm_x(X_test_raw)

    y_train = norm_y(y_train_raw)

    best_sigma = None
    best_r2_val = -np.inf

    # -----------------------------
    # 参数搜索
    # 训练在归一化 y 空间完成
    # 选择最优 sigma 时，在原始物理量空间比较验证集 R²
    # -----------------------------
    for s in sigma_grid:
        g = GRNN(sigma=s)
        g.fit(X_train, y_train)

        y_val_pred_norm = g.predict(X_val)
        y_val_pred = denorm_y(y_val_pred_norm)

        r2_val = r2_score(y_val_raw, y_val_pred)
        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_sigma = s

    print(f"GRNN 最优 sigma = {best_sigma}, val R² = {best_r2_val:.6f}")

    # -----------------------------
    # 用 train + val 训练最终模型
    # 但归一化统计量仍固定使用 train
    # -----------------------------
    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])

    X_train_val = norm_x(X_train_val_raw)
    y_train_val = norm_y(y_train_val_raw)

    g_final = GRNN(sigma=best_sigma)
    g_final.fit(X_train_val, y_train_val)

    y_pred_test_norm = g_final.predict(X_test)
    y_pred_test = denorm_y(y_pred_test_norm)

    # -----------------------------
    # 指标在原始物理量空间计算
    # -----------------------------
    grnn_r2 = r2_score(y_test_raw, y_pred_test)
    grnn_are = average_relative_error(y_test_raw, y_pred_test)

    print("\n===== GRNN 结果 =====")
    print(f"R²  = {grnn_r2:.6f}")
    print(f"ARE = {grnn_are:.6f} %")

    df_out = pd.DataFrame({
        "true": y_test_raw,
        "GRNN_pred": y_pred_test,
    })
    df_out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n预测文件已保存：{save_csv_path}")

    return {
        "r2": grnn_r2,
        "are": grnn_are,
        "best_sigma": best_sigma,
        "y_true": y_test_raw,
        "y_pred": y_pred_test,
        "norm_stats": {
            "X_min": X_min,
            "X_max": X_max,
            "y_min": y_min,
            "y_max": y_max,
        },
    }


if __name__ == "__main__":
    train_and_eval_grnn()