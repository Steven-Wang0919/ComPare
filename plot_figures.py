# -*- coding: utf-8 -*-
"""
生成论文所需图像（基于 compare_all.py 已保存的 CSV）

输入：
- ./output_data/model_metrics.csv
- ./output_data/model_predictions.csv

输出（保存到 ./output_picture/）：
- r2_comparison_zoomed.png
- mrs_comparison_zoomed.png
- true_vs_predicted.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data_dir = "output_data"
    pic_dir = "output_picture"
    os.makedirs(pic_dir, exist_ok=True)

    metrics_path = os.path.join(data_dir, "model_metrics.csv")
    pred_path = os.path.join(data_dir, "model_predictions.csv")

    if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
        raise FileNotFoundError(
            "找不到模型结果文件，请先运行 compare_all.py 生成 "
            "model_metrics.csv 和 model_predictions.csv。"
        )

    # 1. 读取指标表
    df_metrics = pd.read_csv(metrics_path)
    models = df_metrics["Model"].values
    r2_vals = df_metrics["R2"].values
    mrs_vals = df_metrics["MRS(%)"].values

    # 2. 读取预测表
    df_pred = pd.read_csv(pred_path)
    y_true = df_pred["true"].values
    mlp_pred = df_pred["MLP_pred"].values
    grnn_pred = df_pred["GRNN_pred"].values
    kan_pred = df_pred["KAN_pred"].values

    # ----------------------
    # 图 1: R² 对比（缩放 + 数值标签）
    # ----------------------
    plt.figure()
    plt.bar(models, r2_vals)
    plt.ylabel("R²")
    plt.title("R² Comparison (Zoomed)")

    min_r2 = float(np.min(r2_vals))
    # 稍微向下留一点空间，让差异更明显
    plt.ylim(min_r2 - 0.01, 1.0)

    for i, v in enumerate(r2_vals):
        plt.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=10)

    r2_fig_path = os.path.join(pic_dir, "r2_comparison_zoomed.png")
    plt.savefig(r2_fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"R² 缩放对比图已保存：{r2_fig_path}")

    # ----------------------
    # 图 2: MRS 对比（缩放 + 数值标签）
    # ----------------------
    plt.figure()
    plt.bar(models, mrs_vals)
    plt.ylabel("MRS (%)")
    plt.title("MRS Comparison (Zoomed)")

    min_mrs = float(np.min(mrs_vals))
    max_mrs = float(np.max(mrs_vals))
    margin = (max_mrs - min_mrs) * 0.2 if max_mrs > min_mrs else 1.0
    plt.ylim(max(0.0, min_mrs - margin), max_mrs + margin)

    for i, v in enumerate(mrs_vals):
        plt.text(i, v + margin * 0.05, f"{v:.2f}", ha="center", fontsize=10)

    mrs_fig_path = os.path.join(pic_dir, "mrs_comparison_zoomed.png")
    plt.savefig(mrs_fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"MRS 缩放对比图已保存：{mrs_fig_path}")

    # ----------------------
    # 图 3: 真值 vs 预测散点图
    # ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    model_preds = [
        ("MLP", mlp_pred),
        ("GRNN", grnn_pred),
        ("KAN", kan_pred),
    ]

    y_min = float(np.min(y_true))
    y_max = float(np.max(y_true))
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    y_min_plot = y_min - pad
    y_max_plot = y_max + pad

    for ax, (name, pred) in zip(axes, model_preds):
        ax.scatter(y_true, pred, alpha=0.7)
        ax.plot([y_min_plot, y_max_plot], [y_min_plot, y_max_plot], "k--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    fig.suptitle("True vs Predicted")
    plt.tight_layout()
    tvp_fig_path = os.path.join(pic_dir, "true_vs_predicted.png")
    fig.savefig(tvp_fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"真值-预测散点图已保存：{tvp_fig_path}")


if __name__ == "__main__":
    main()
