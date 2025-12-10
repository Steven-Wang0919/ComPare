# -*- coding: utf-8 -*-
"""
生成论文所需的高质量图像（优化版）

输入：
- ./output_data/model_metrics.csv
- ./output_data/model_predictions.csv

输出（保存到 ./output_picture/）：
- r2_comparison_zoomed.png      (R² 对比柱状图)
- mrs_comparison_zoomed.png     (MRS 对比柱状图)
- true_vs_predicted.png         (真值 vs 预测散点图，带 y=x 参考线)
- residuals_distribution.png    (残差分布直方图，新增)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局绘图风格（学术风）
try:
    # 尝试使用 seaborn 的 paper 风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
except AttributeError:
    # 兼容旧版 seaborn
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

# 如果没有中文字体，可以尝试设置（可选，视您的环境而定）
plt.rcParams['axes.unicode_minus'] = False


def main():
    data_dir = "output_data"
    pic_dir = "output_picture"
    os.makedirs(pic_dir, exist_ok=True)

    metrics_path = os.path.join(data_dir, "model_metrics.csv")
    pred_path = os.path.join(data_dir, "model_predictions.csv")

    if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
        raise FileNotFoundError("找不到数据文件，请先运行 compare_all.py。")

    # 读取数据
    df_metrics = pd.read_csv(metrics_path)
    df_pred = pd.read_csv(pred_path)

    models = df_metrics["Model"].values
    r2_vals = df_metrics["R2"].values
    mrs_vals = df_metrics["MRS(%)"].values

    # 定义颜色方案 (蓝色系，沉稳)
    colors = sns.color_palette("viridis", n_colors=len(models))

    # ==========================================
    # 1. R² 对比图 (Zoomed)
    # ==========================================
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, r2_vals, color=colors, alpha=0.8, width=0.6, edgecolor='black')

    # 动态设置 Y 轴范围，凸显差异
    min_r2 = min(r2_vals)
    plt.ylim(min_r2 - (1 - min_r2) * 0.2, 1.005)

    plt.ylabel("R-squared ($R^2$)", fontsize=12)
    plt.title("Model Comparison: $R^2$ Score", fontsize=14, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0002,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "r2_comparison_zoomed.png"), dpi=300)
    plt.close()
    print("图表已保存: r2_comparison_zoomed.png")

    # ==========================================
    # 2. MRS 对比图
    # ==========================================
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, mrs_vals, color=colors, alpha=0.8, width=0.6, edgecolor='black')

    # 设置 Y 轴
    max_mrs = max(mrs_vals)
    plt.ylim(0, max_mrs * 1.2)

    plt.ylabel("Mean Relative Error (%)", fontsize=12)
    plt.title("Model Comparison: MRS", fontsize=14, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + max_mrs * 0.02,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "mrs_comparison_zoomed.png"), dpi=300)
    plt.close()
    print("图表已保存: mrs_comparison_zoomed.png")

    # ==========================================
    # 3. 真值 vs 预测散点图 (带 y=x 参考线)
    # ==========================================
    y_true = df_pred["true"].values
    model_cols = ["MLP_pred", "GRNN_pred", "KAN_pred"]
    model_names = ["MLP", "GRNN", "KAN"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # 计算统一的坐标轴范围，保证正方形比例
    all_vals = np.concatenate([y_true] + [df_pred[c].values for c in model_cols])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = (val_max - val_min) * 0.05
    axis_lim = [val_min - margin, val_max + margin]

    for i, (ax, col, name) in enumerate(zip(axes, model_cols, model_names)):
        y_pred = df_pred[col].values

        # 绘制 y=x 参考线
        ax.plot(axis_lim, axis_lim, color='red', linestyle='--', linewidth=1.5, label='Ideal ($y=x$)')

        # 绘制散点
        ax.scatter(y_true, y_pred, alpha=0.6, s=40, color=colors[i], edgecolor='w', label='Samples')

        # 计算该子图的 R2 标注在图上
        r2 = r2_vals[i]  # 假设顺序一致
        ax.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=ax.transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel("True Value (g/min)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Predicted Value (g/min)", fontsize=12)

        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "true_vs_predicted.png"), dpi=300)
    plt.close()
    print("图表已保存: true_vs_predicted.png")

    # ==========================================
    # 4. (新增) 残差分布直方图
    # ==========================================
    # 残差 = 预测 - 真实。 理想情况下应服从以 0 为均值的正态分布。
    plt.figure(figsize=(10, 5))

    bins = 15
    for i, (col, name) in enumerate(zip(model_cols, model_names)):
        residuals = df_pred[col].values - y_true
        sns.histplot(residuals, bins=bins, kde=True, element="step",
                     label=name, color=colors[i], alpha=0.3)

    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Residuals Distribution ($y_{pred} - y_{true}$)", fontsize=14)
    plt.xlabel("Prediction Error (g/min)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "residuals_distribution.png"), dpi=300)
    plt.close()
    print("图表已保存: residuals_distribution.png")

    print("\n所有绘图完成！图片保存在 output_picture 文件夹中。")


if __name__ == "__main__":
    main()