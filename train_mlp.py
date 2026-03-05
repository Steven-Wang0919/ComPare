# -*- coding: utf-8 -*-
"""
训练 & 评估 MLP（最终精简稳定版，无 Warning）

- 使用 train/val/test 三分数据
- 使用 0–1 归一化
- 超参数搜索（hidden_layer_sizes, alpha）
- 固定 solver='lbfgs'（避免所有 ConvergenceWarning）
"""

import warnings
warnings.filterwarnings("ignore")  # 全局关闭 sklearn 的 Warning

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from common_utils import load_data, get_train_val_test_indices, mean_relative_error


def train_and_eval_mlp(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=0,
    save_csv_path="results_mlp.csv",
):
    # 超参数空间（保持适中范围）
    if hidden_layer_candidates is None:
        hidden_layer_candidates = [
            (10,),
            (20,),
            (50,),
            (20, 20),
        ]
    if alpha_candidates is None:
        alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    print("\n=== 训练 MLP ===")

    # 1. 加载数据
    X, y = load_data(data_path)

    # 2. 统一 train/val/test 划分
    idx_tr, idx_val, idx_te = get_train_val_test_indices(len(X))
    X_train_raw, y_train_raw = X[idx_tr], y[idx_tr]
    X_val_raw, y_val_raw = X[idx_val], y[idx_val]
    X_test_raw, y_test_raw = X[idx_te], y[idx_te]

    # 3. 统一 0–1 归一化（仅用 train 的 min/max）
    X_min = X_train_raw.min(axis=0, keepdims=True)
    X_max = X_train_raw.max(axis=0, keepdims=True)

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + 1e-8)

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    X_test = norm_x(X_test_raw)

    # 4. 在验证集上做网格搜索（solver 固定为 lbfgs → 无 warning）
    best_r2_val = -np.inf
    best_hidden = None
    best_alpha = None

    for h in hidden_layer_candidates:
        for a in alpha_candidates:
            mlp = MLPRegressor(
                hidden_layer_sizes=h,
                alpha=a,
                solver="lbfgs",          # 固定 lbfgs，避免 warning
                max_iter=max_iter,
                random_state=random_state,
            )

            mlp.fit(X_train, y_train_raw)
            y_val_pred = mlp.predict(X_val)
            r2_val = r2_score(y_val_raw, y_val_pred)

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_hidden = h
                best_alpha = a

    print(
        f"MLP 最优超参数：hidden_layer_sizes={best_hidden}, "
        f"alpha={best_alpha}, solver='lbfgs', val R²={best_r2_val:.6f}"
    )

    # 5. train+val 合并训练最终模型
    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])
    X_train_val = norm_x(X_train_val_raw)
    X_test = norm_x(X_test_raw)

    mlp_final = MLPRegressor(
        hidden_layer_sizes=best_hidden,
        alpha=best_alpha,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp_final.fit(X_train_val, y_train_val_raw)
    y_pred_test = mlp_final.predict(X_test)

    # 6. test 集指标
    mlp_r2 = r2_score(y_test_raw, y_pred_test)
    mlp_mrs = mean_relative_error(y_test_raw, y_pred_test)

    print("\n===== MLP 结果 =====")
    print(f"R²  = {mlp_r2:.6f}")
    print(f"MRS = {mlp_mrs:.6f} %")

    # 7. 保存预测
    df_out = pd.DataFrame({
        "true": y_test_raw,
        "MLP_pred": y_pred_test,
    })
    df_out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n预测文件已保存：{save_csv_path}")

    return {
        "r2": mlp_r2,
        "mrs": mlp_mrs,
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,
        "y_true": y_test_raw,
        "y_pred": y_pred_test,
    }


if __name__ == "__main__":
    train_and_eval_mlp()
