# -*- coding: utf-8 -*-
"""
train_mlp.py

正向 MLP：
- 统一 train/val/test 划分
- 统一 train-only normalization
- 默认不再写仓库根目录 results_mlp.csv
- 独立运行时输出到 runs/<timestamp>_train_mlp/
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from common_utils import load_data, get_train_val_test_indices, average_relative_error
from run_utils import append_manifest_outputs, create_run_dir, save_dataframe, write_manifest


EPS = 1e-8


def train_and_eval_mlp(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=0,
    save_csv_path=None,
):
    if hidden_layer_candidates is None:
        hidden_layer_candidates = [(10,), (20,), (50,), (20, 20)]
    if alpha_candidates is None:
        alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    print("\n=== 训练 MLP（统一 y 归一化口径） ===")

    X, y = load_data(data_path)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X, y=y)

    X_train_raw, y_train_raw = X[idx_tr], y[idx_tr]
    X_val_raw, y_val_raw = X[idx_val], y[idx_val]
    X_test_raw, y_test_raw = X[idx_te], y[idx_te]

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

    best_r2_val = -np.inf
    best_hidden = None
    best_alpha = None

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

    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])

    X_train_val = norm_x(X_train_val_raw)
    y_train_val = norm_y(y_train_val_raw)

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

    mlp_r2 = r2_score(y_test_raw, y_pred_test)
    mlp_are = average_relative_error(y_test_raw, y_pred_test)

    print("\n===== MLP 结果 =====")
    print(f"R²  = {mlp_r2:.6f}")
    print(f"ARE = {mlp_are:.6f} %")

    if save_csv_path is not None:
        df_out = pd.DataFrame({
            "true": y_test_raw,
            "MLP_pred": y_pred_test,
        })
        save_dataframe(df_out, save_csv_path)
        print(f"预测文件已保存：{save_csv_path}")

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


def main():
    run_dir = create_run_dir("train_mlp")
    output_csv = os.path.join(run_dir, "results_mlp.csv")

    write_manifest(
        run_dir,
        script_name="train_mlp.py",
        data_path="data/dataset.xlsx",
        seed=0,
        params={
            "hidden_layer_candidates": [(10,), (20,), (50,), (20, 20)],
            "alpha_candidates": [1e-6, 1e-5, 1e-4, 1e-3],
            "max_iter": 5000,
            "random_state": 0,
        },
    )

    train_and_eval_mlp(save_csv_path=output_csv)

    append_manifest_outputs(
        run_dir,
        [{"path": "results_mlp.csv"}],
    )

    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()