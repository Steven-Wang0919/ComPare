# -*- coding: utf-8 -*-
"""
训练 & 评估 GRNN

- 统一使用 train/val/test 三分数据
- 统一使用 0–1 归一化（与 KAN 相同）
- 在验证集上选择最优 σ
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, mean_relative_error


# =========================
# GRNN 模型
# =========================
class GRNN:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def _predict_one(self, x):
        diff = self.X - x
        dist2 = np.sum(diff ** 2, axis=1)
        w = np.exp(-dist2 / (2 * self.sigma ** 2))
        if w.sum() == 0:
            return self.y.mean()
        return np.sum(w * self.y) / w.sum()

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


def train_and_eval_grnn(
    data_path="data/数据集.xlsx",
    sigma_grid=None,
    save_csv_path="results_grnn.csv",
):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.1, 4.0, 40)

    # 1. 加载数据
    X, y = load_data(data_path)

    # 2. 统一 train/val/test 划分
    idx_tr, idx_val, idx_te = get_train_val_test_indices(len(X))
    X_train_raw, y_train_raw = X[idx_tr], y[idx_tr]
    X_val_raw, y_val_raw = X[idx_val], y[idx_val]
    X_test_raw, y_test_raw = X[idx_te], y[idx_te]

    print("\n=== 训练 GRNN ===")

    # 3. 统一 0–1 归一化（仅用 train 的 min/max）
    X_min = X_train_raw.min(axis=0, keepdims=True)
    X_max = X_train_raw.max(axis=0, keepdims=True)

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + 1e-8)

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    X_test = norm_x(X_test_raw)

    # 4. 在验证集上选最佳 σ
    best_sigma = None
    best_r2_val = -np.inf

    for s in sigma_grid:
        g = GRNN(sigma=s)
        g.fit(X_train, y_train_raw)
        y_val_pred = g.predict(X_val)
        r2_val = r2_score(y_val_raw, y_val_pred)
        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_sigma = s

    print(f"GRNN 最优 σ = {best_sigma}, val R² = {best_r2_val:.6f}")

    # 5. 用 train+val 重新训练，并在 test 上评估
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train_raw, y_val_raw])

    g_final = GRNN(sigma=best_sigma)
    g_final.fit(X_train_val, y_train_val)

    X_test = norm_x(X_test_raw)  # 仍用 train 的 min/max
    y_pred_test = g_final.predict(X_test)

    grnn_r2 = r2_score(y_test_raw, y_pred_test)
    grnn_mrs = mean_relative_error(y_test_raw, y_pred_test)

    print("\n===== GRNN 结果 =====")
    print(f"R²  = {grnn_r2:.6f}")
    print(f"MRS = {grnn_mrs:.6f} %")

    # 6. 保存预测
    df_out = pd.DataFrame({
        "true": y_test_raw,
        "GRNN_pred": y_pred_test,
    })
    df_out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n预测文件已保存：{save_csv_path}")

    return {
        "r2": grnn_r2,
        "mrs": grnn_mrs,
        "best_sigma": best_sigma,
        "y_true": y_test_raw,
        "y_pred": y_pred_test,
    }


if __name__ == "__main__":
    train_and_eval_grnn()
