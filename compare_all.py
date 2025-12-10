# -*- coding: utf-8 -*-
"""
综合对比三个模型（MLP / GRNN / KAN）

职责：
- 调用各自的 train_and_eval_* 函数
- 将指标表保存到： ./output_data/model_metrics.csv
- 将统一的真值 & 各模型预测保存到： ./output_data/model_predictions.csv

不画任何图，画图交给 plot_figures.py
"""

import os
import numpy as np
import pandas as pd

from train_mlp import train_and_eval_mlp
from train_grnn import train_and_eval_grnn
from train_kan import train_and_eval_kan


def main():
    # 1. 创建输出文件夹
    data_dir = "output_data"
    os.makedirs(data_dir, exist_ok=True)

    # 2. 依次训练并评估三个模型
    mlp_res = train_and_eval_mlp()
    grnn_res = train_and_eval_grnn()
    kan_res = train_and_eval_kan()

    # 3. 构建指标表
    metrics = [
        {
            "Model": "MLP",
            "R2": mlp_res["r2"],
            "MRS(%)": mlp_res["mrs"],
            "Hyperparams": f"hidden={mlp_res.get('best_hidden')}, "
                           f"alpha={mlp_res.get('best_alpha')}",
        },
        {
            "Model": "GRNN",
            "R2": grnn_res["r2"],
            "MRS(%)": grnn_res["mrs"],
            "Hyperparams": f"sigma={grnn_res.get('best_sigma')}",
        },
        {
            "Model": "KAN",
            "R2": kan_res["r2"],
            "MRS(%)": kan_res["mrs"],
            "Hyperparams": f"hidden={kan_res.get('best_hidden_dim')}, "
                           f"lr={kan_res.get('best_lr')}, "
                           f"wd={kan_res.get('best_weight_decay')}",
        },
    ]

    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(data_dir, "model_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n模型指标表已保存：{metrics_path}")

    # 4. 统一真值和预测，保存到一个表
    # 理论上三个模型的 y_true 应该完全相同，这里做一个长度保护
    y_true_mlp = np.asarray(mlp_res["y_true"]).reshape(-1)
    y_true_grnn = np.asarray(grnn_res["y_true"]).reshape(-1)
    y_true_kan = np.asarray(kan_res["y_true"]).reshape(-1)

    min_len = min(len(y_true_mlp), len(y_true_grnn), len(y_true_kan))

    y_true = y_true_mlp[:min_len]
    mlp_pred = np.asarray(mlp_res["y_pred"]).reshape(-1)[:min_len]
    grnn_pred = np.asarray(grnn_res["y_pred"]).reshape(-1)[:min_len]
    kan_pred = np.asarray(kan_res["y_pred"]).reshape(-1)[:min_len]

    df_pred = pd.DataFrame({
        "true": y_true,
        "MLP_pred": mlp_pred,
        "GRNN_pred": grnn_pred,
        "KAN_pred": kan_pred,
    })

    pred_path = os.path.join(data_dir, "model_predictions.csv")
    df_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"预测对比表已保存：{pred_path}")

    print("\n对比完成，可运行 plot_figures.py 生成图像。")


if __name__ == "__main__":
    main()
