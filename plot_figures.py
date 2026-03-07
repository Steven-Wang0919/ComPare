# -*- coding: utf-8 -*-
"""
生成论文所需的高质量图像（ARE 统一版）

输入：
- ./output_data/model_metrics.csv
- ./output_data/model_predictions.csv

输出（保存到 ./output_picture/）：
- r2_comparison_zoomed.png      (R² 对比柱状图)
- are_comparison_zoomed.png     (ARE 对比柱状图)
- true_vs_predicted.png         (真值 vs 预测散点图，带 y=x 参考线)
- residuals_distribution.png    (残差分布直方图)

改进点：
1. 不再依赖 model_metrics.csv 的行顺序
2. 按模型名显式映射 R² / ARE / 颜色
3. 对输入列做严格校验，避免静默出错
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局绘图风格（学术风）
try:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
except AttributeError:
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

plt.rcParams["axes.unicode_minus"] = False

EXPECTED_MODELS = ["MLP", "GRNN", "KAN"]
PRED_COL_MAP = {
    "MLP": "MLP_pred",
    "GRNN": "GRNN_pred",
    "KAN": "KAN_pred",
}


def _validate_required_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} 缺少必要列: {missing}。\n"
            f"当前列为: {list(df.columns)}"
        )


def _build_metric_map(df_metrics):
    required_cols = ["Model", "R2", "ARE(%)"]
    _validate_required_columns(df_metrics, required_cols, "model_metrics.csv")

    model_set = set(df_metrics["Model"].astype(str).values.tolist())
    expected_set = set(EXPECTED_MODELS)

    missing_models = expected_set - model_set
    extra_models = model_set - expected_set

    if missing_models:
        raise ValueError(
            f"model_metrics.csv 缺少模型: {sorted(missing_models)}。"
        )

    if extra_models:
        print(f"警告：model_metrics.csv 中存在未使用的额外模型: {sorted(extra_models)}")

    duplicated = df_metrics["Model"][df_metrics["Model"].duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"model_metrics.csv 中存在重复模型记录: {duplicated}。"
        )

    r2_map = dict(zip(df_metrics["Model"], df_metrics["R2"]))
    are_map = dict(zip(df_metrics["Model"], df_metrics["ARE(%)"]))

    return r2_map, are_map


def _build_color_map():
    palette = sns.color_palette("viridis", n_colors=len(EXPECTED_MODELS))
    return {model: color for model, color in zip(EXPECTED_MODELS, palette)}


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

    # 校验预测表字段
    required_pred_cols = ["true"] + [PRED_COL_MAP[m] for m in EXPECTED_MODELS]
    _validate_required_columns(df_pred, required_pred_cols, "model_predictions.csv")

    # 建立显式映射，不依赖行顺序
    r2_map, are_map = _build_metric_map(df_metrics)
    color_map = _build_color_map()

    models = EXPECTED_MODELS
    r2_vals = [float(r2_map[m]) for m in models]
    are_vals = [float(are_map[m]) for m in models]
    colors = [color_map[m] for m in models]

    # ==========================================
    # 1. R² 对比图
    # ==========================================
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, r2_vals, color=colors, alpha=0.8, width=0.6, edgecolor="black")

    min_r2 = min(r2_vals)
    plt.ylim(min_r2 - (1 - min_r2) * 0.2, 1.005)

    plt.ylabel("R-squared ($R^2$)", fontsize=12)
    plt.title("Model Comparison: $R^2$ Score", fontsize=14, pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0002,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "r2_comparison_zoomed.png"), dpi=300)
    plt.close()
    print("图表已保存: r2_comparison_zoomed.png")

    # ==========================================
    # 2. ARE 对比图
    # ==========================================
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, are_vals, color=colors, alpha=0.8, width=0.6, edgecolor="black")

    max_are = max(are_vals)
    plt.ylim(0, max_are * 1.2)

    plt.ylabel("Average Relative Error (%)", fontsize=12)
    plt.title("Model Comparison: ARE", fontsize=14, pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_are * 0.02,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "are_comparison_zoomed.png"), dpi=300)
    plt.close()
    print("图表已保存: are_comparison_zoomed.png")

    # ==========================================
    # 3. 真值 vs 预测散点图
    # ==========================================
    y_true = df_pred["true"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    all_vals = np.concatenate([y_true] + [df_pred[PRED_COL_MAP[m]].values for m in models])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = (val_max - val_min) * 0.05
    axis_lim = [val_min - margin, val_max + margin]

    for i, (ax, model_name) in enumerate(zip(axes, models)):
        pred_col = PRED_COL_MAP[model_name]
        y_pred = df_pred[pred_col].values
        color = color_map[model_name]
        r2 = float(r2_map[model_name])

        ax.plot(axis_lim, axis_lim, color="red", linestyle="--", linewidth=1.5, label="Ideal ($y=x$)")
        ax.scatter(y_true, y_pred, alpha=0.6, s=40, color=color, edgecolor="w", label="Samples")

        ax.text(
            0.05,
            0.95,
            f"$R^2 = {r2:.4f}$",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("True Value (g/min)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Predicted Value (g/min)", fontsize=12)

        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "true_vs_predicted.png"), dpi=300)
    plt.close()
    print("图表已保存: true_vs_predicted.png")

    # ==========================================
    # 4. 残差分布直方图
    # ==========================================
    plt.figure(figsize=(10, 5))

    bins = 15
    for model_name in models:
        pred_col = PRED_COL_MAP[model_name]
        residuals = df_pred[pred_col].values - y_true
        sns.histplot(
            residuals,
            bins=bins,
            kde=True,
            element="step",
            label=model_name,
            color=color_map[model_name],
            alpha=0.3,
        )

    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residuals Distribution ($y_{pred} - y_{true}$)", fontsize=14)
    plt.xlabel("Prediction Error (g/min)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "residuals_distribution.png"), dpi=300)
    plt.close()
    print("图表已保存: residuals_distribution.png")

    print("\n所有绘图完成！图片保存在 output_picture 文件夹中。")


if __name__ == "__main__":
    main()