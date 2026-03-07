# -*- coding: utf-8 -*-
"""
训练 & 评估 MLP（统一口径版）

修订说明：
- 使用 train/val/test 三分数据
- X 使用 train 统计量做 0–1 归一化
- y 也使用 train 统计量做 0–1 归一化（与 KAN 口径一致）
- 超参数搜索（hidden_layer_sizes, alpha）
- 固定 solver='lbfgs'
- 选择最优超参数时，验证集指标仍在“原始物理量空间”上计算
- 最终测试指标统一使用平均相对误差（ARE）和 R²（原始物理量空间）
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from common_utils import load_data, get_train_val_test_indices, average_relative_error


EPS = 1e-8


def train_and_eval_mlp(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=0,
    save_csv_path="results_mlp.csv",
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

    print("\n=== 训练 MLP（统一 y 归一化口径） ===")

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
    y_val = norm_y(y_val_raw)   # 这里只是保留口径，选择模型时仍用原始量评估
    _ = y_val  # 防止后续静态检查误报未使用

    best_r2_val = -np.inf
    best_hidden = None
    best_alpha = None

    # -----------------------------
    # 超参数搜索
    # 训练时用归一化 y
    # 选择时在原始物理量空间比较 R²
    # -----------------------------
    for h in hidden_layer_candidates:
        for a in alpha_candidates:
            mlp = MLPRegressor(
                hidden_layer_sizes=h,
                alpha=a,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )

            mlp.fit(X_train, y_train)

            y_val_pred_norm = mlp.predict(X_val)
            y_val_pred = denorm_y(y_val_pred_norm)

            r2_val = r2_score(y_val_raw, y_val_pred)

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_hidden = h
                best_alpha = a

    print(
        f"MLP 最优超参数：hidden_layer_sizes={best_hidden}, "
        f"alpha={best_alpha}, solver='lbfgs', val R²={best_r2_val:.6f}"
    )

    # -----------------------------
    # 用 train + val 训练最终模型
    # 但归一化统计量仍然固定使用 train
    # 避免和既定实验协议不一致
    # -----------------------------
    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])

    X_train_val = norm_x(X_train_val_raw)
    y_train_val = norm_y(y_train_val_raw)
    X_test = norm_x(X_test_raw)

    mlp_final = MLPRegressor(
        hidden_layer_sizes=best_hidden,
        alpha=best_alpha,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp_final.fit(X_train_val, y_train_val)

    y_pred_test_norm = mlp_final.predict(X_test)
    y_pred_test = denorm_y(y_pred_test_norm)

    # -----------------------------
    # 指标在原始物理量空间计算
    # -----------------------------
    mlp_r2 = r2_score(y_test_raw, y_pred_test)
    mlp_are = average_relative_error(y_test_raw, y_pred_test)

    print("\n===== MLP 结果 =====")
    print(f"R²  = {mlp_r2:.6f}")
    print(f"ARE = {mlp_are:.6f} %")

    df_out = pd.DataFrame({
        "true": y_test_raw,
        "MLP_pred": y_pred_test,
    })
    df_out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n预测文件已保存：{save_csv_path}")

    return {
        "r2": mlp_r2,
        "are": mlp_are,
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,
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
    train_and_eval_mlp()