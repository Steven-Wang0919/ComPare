# -*- coding: utf-8 -*-
"""
生成论文所需的高质量图像（联动兼容版）

输入：
1. 正向任务（自动兼容两套命名）
   - ./output_data/forward_model_metrics.csv
   - ./output_data/forward_model_predictions.csv
   - 若不存在，则回退到旧命名：
     - ./output_data/model_metrics.csv
     - ./output_data/model_predictions.csv

2. 反向任务（若存在则自动绘图）
   - ./output_data/inverse_model_metrics.csv
   - ./output_data/inverse_model_predictions_all.csv
   - ./output_data/inverse_model_predictions_main.csv

输出（保存到 ./output_picture/）：
正向：
- r2_comparison_zoomed.png
- are_comparison_zoomed.png
- true_vs_predicted.png
- residuals_distribution.png

反向（若检测到反向文件）：
- inverse_r2_main_vs_all.png
- inverse_are_main_vs_all.png
- inverse_true_vs_predicted_main.png
- inverse_true_vs_predicted_all.png

改进点：
1. 自动兼容 compare_all.py 当前输出文件名与历史文件名
2. 不再依赖 metrics.csv 的行顺序
3. 按模型名显式映射指标 / 颜色 / 预测列
4. 对输入列做严格校验，避免静默出错
5. 兼容结构修复后的反向 KAN 命名：inverse_KAN_repaired
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
except AttributeError:
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

plt.rcParams["axes.unicode_minus"] = False

FORWARD_MODELS = ["MLP", "GRNN", "KAN"]
FORWARD_PRED_COL_MAP = {
    "MLP": "MLP_pred",
    "GRNN": "GRNN_pred",
    "KAN": "KAN_pred",
}

INVERSE_MODELS = ["inverse_MLP", "inverse_GRNN", "inverse_KAN_repaired"]
INVERSE_DISPLAY_MAP = {
    "inverse_MLP": "inverse_MLP",
    "inverse_GRNN": "inverse_GRNN",
    "inverse_KAN_repaired": "inverse_KAN\n(repaired)",
}
INVERSE_PRED_COL_MAP = {
    "inverse_MLP": "inverse_MLP_pred",
    "inverse_GRNN": "inverse_GRNN_pred",
    "inverse_KAN_repaired": "inverse_KAN_repaired_pred",
}


def _validate_required_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} 缺少必要列: {missing}.\n当前列为: {list(df.columns)}"
        )


def _resolve_input_path(data_dir, preferred_name, legacy_name=None):
    preferred_path = os.path.join(data_dir, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path

    if legacy_name is not None:
        legacy_path = os.path.join(data_dir, legacy_name)
        if os.path.exists(legacy_path):
            return legacy_path
        raise FileNotFoundError(
            f"找不到数据文件: {preferred_path} 或 {legacy_path}。请先运行 compare_all.py。"
        )

    raise FileNotFoundError(
        f"找不到数据文件: {preferred_path}。请先运行 compare_all.py。"
    )


def _build_color_map(models):
    palette = sns.color_palette("viridis", n_colors=len(models))
    return {model: color for model, color in zip(models, palette)}


def _build_metric_map(df_metrics, model_col, required_metric_cols, expected_models, df_name):
    required_cols = [model_col] + required_metric_cols
    _validate_required_columns(df_metrics, required_cols, df_name)

    model_set = set(df_metrics[model_col].astype(str).values.tolist())
    expected_set = set(expected_models)

    missing_models = expected_set - model_set
    extra_models = model_set - expected_set

    if missing_models:
        raise ValueError(f"{df_name} 缺少模型: {sorted(missing_models)}。")

    if extra_models:
        print(f"警告：{df_name} 中存在未使用的额外模型: {sorted(extra_models)}")

    duplicated = df_metrics[model_col][df_metrics[model_col].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{df_name} 中存在重复模型记录: {duplicated}。")

    metric_maps = []
    for col in required_metric_cols:
        metric_maps.append(dict(zip(df_metrics[model_col], df_metrics[col])))
    return metric_maps


def _save_forward_plots(data_dir, pic_dir):
    metrics_path = _resolve_input_path(
        data_dir, "forward_model_metrics.csv", "model_metrics.csv"
    )
    pred_path = _resolve_input_path(
        data_dir, "forward_model_predictions.csv", "model_predictions.csv"
    )

    print(f"读取正向指标文件: {metrics_path}")
    print(f"读取正向预测文件: {pred_path}")

    df_metrics = pd.read_csv(metrics_path)
    df_pred = pd.read_csv(pred_path)

    required_pred_cols = ["true"] + [FORWARD_PRED_COL_MAP[m] for m in FORWARD_MODELS]
    _validate_required_columns(df_pred, required_pred_cols, os.path.basename(pred_path))

    r2_map, are_map = _build_metric_map(
        df_metrics=df_metrics,
        model_col="Model",
        required_metric_cols=["R2", "ARE(%)"],
        expected_models=FORWARD_MODELS,
        df_name=os.path.basename(metrics_path),
    )

    color_map = _build_color_map(FORWARD_MODELS)
    models = FORWARD_MODELS
    r2_vals = [float(r2_map[m]) for m in models]
    are_vals = [float(are_map[m]) for m in models]
    colors = [color_map[m] for m in models]

    # 1. R² 对比图
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, r2_vals, color=colors, alpha=0.8, width=0.6, edgecolor="black")

    min_r2 = min(r2_vals)
    plt.ylim(min_r2 - (1 - min_r2) * 0.2, 1.005)
    plt.ylabel("R-squared ($R^2$)", fontsize=12)
    plt.title("Forward Model Comparison: $R^2$", fontsize=14, pad=15)
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

    # 2. ARE 对比图
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, are_vals, color=colors, alpha=0.8, width=0.6, edgecolor="black")

    max_are = max(are_vals)
    plt.ylim(0, max_are * 1.2)
    plt.ylabel("Average Relative Error (%)", fontsize=12)
    plt.title("Forward Model Comparison: ARE", fontsize=14, pad=15)
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

    # 3. 真值 vs 预测散点图
    y_true = df_pred["true"].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    all_vals = np.concatenate([y_true] + [df_pred[FORWARD_PRED_COL_MAP[m]].values for m in models])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = (val_max - val_min) * 0.05
    axis_lim = [val_min - margin, val_max + margin]

    for i, (ax, model_name) in enumerate(zip(axes, models)):
        pred_col = FORWARD_PRED_COL_MAP[model_name]
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

    # 4. 残差分布
    plt.figure(figsize=(10, 5))
    bins = 15
    for model_name in models:
        pred_col = FORWARD_PRED_COL_MAP[model_name]
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
    plt.title("Forward Residuals Distribution ($y_{pred} - y_{true}$)", fontsize=14)
    plt.xlabel("Prediction Error (g/min)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "residuals_distribution.png"), dpi=300)
    plt.close()
    print("图表已保存: residuals_distribution.png")


def _save_inverse_metric_barplot(df_metrics, metric_main_col, metric_all_col, ylabel, title, out_path):
    color_map = _build_color_map(INVERSE_MODELS)
    display_labels = [INVERSE_DISPLAY_MAP[m] for m in INVERSE_MODELS]

    main_map, all_map = _build_metric_map(
        df_metrics=df_metrics,
        model_col="Model",
        required_metric_cols=[metric_main_col, metric_all_col],
        expected_models=INVERSE_MODELS,
        df_name="inverse_model_metrics.csv",
    )

    x = np.arange(len(INVERSE_MODELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2,
        [float(main_map[m]) for m in INVERSE_MODELS],
        width,
        label="Main subset",
        color=[color_map[m] for m in INVERSE_MODELS],
        alpha=0.9,
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        [float(all_map[m]) for m in INVERSE_MODELS],
        width,
        label="All test",
        color=[color_map[m] for m in INVERSE_MODELS],
        alpha=0.45,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    for bars, fmt in [(bars1, "{:.4f}"), (bars2, "{:.4f}")]:
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"图表已保存: {os.path.basename(out_path)}")


def _save_inverse_scatter(df_pred, subset_name, out_path):
    required_cols = ["true_speed"] + [INVERSE_PRED_COL_MAP[m] for m in INVERSE_MODELS]
    _validate_required_columns(df_pred, required_cols, f"inverse predictions ({subset_name})")

    y_true = df_pred["true_speed"].values
    color_map = _build_color_map(INVERSE_MODELS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    all_vals = np.concatenate([y_true] + [df_pred[INVERSE_PRED_COL_MAP[m]].values for m in INVERSE_MODELS])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = (val_max - val_min) * 0.05 if val_max > val_min else 1.0
    axis_lim = [val_min - margin, val_max + margin]

    for i, (ax, model_name) in enumerate(zip(axes, INVERSE_MODELS)):
        pred_col = INVERSE_PRED_COL_MAP[model_name]
        y_pred = df_pred[pred_col].values

        ax.plot(axis_lim, axis_lim, color="red", linestyle="--", linewidth=1.5, label="Ideal ($y=x$)")
        ax.scatter(
            y_true, y_pred, alpha=0.6, s=40,
            color=color_map[model_name], edgecolor="w", label="Samples"
        )

        ax.set_title(INVERSE_DISPLAY_MAP[model_name], fontsize=13, fontweight="bold")
        ax.set_xlabel("True Speed (r/min)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Predicted Speed (r/min)", fontsize=12)
            ax.legend(loc="lower right")

        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(f"Inverse Model Comparison ({subset_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {os.path.basename(out_path)}")


def _maybe_save_inverse_plots(data_dir, pic_dir):
    metrics_path = os.path.join(data_dir, "inverse_model_metrics.csv")
    pred_all_path = os.path.join(data_dir, "inverse_model_predictions_all.csv")
    pred_main_path = os.path.join(data_dir, "inverse_model_predictions_main.csv")

    inverse_files_exist = all(os.path.exists(p) for p in [metrics_path, pred_all_path, pred_main_path])
    if not inverse_files_exist:
        print("未检测到完整的反向结果文件，跳过反向任务绘图。")
        return

    print(f"读取反向指标文件: {metrics_path}")
    print(f"读取反向全测试集预测文件: {pred_all_path}")
    print(f"读取反向主结果子集预测文件: {pred_main_path}")

    df_metrics = pd.read_csv(metrics_path)
    df_pred_all = pd.read_csv(pred_all_path)
    df_pred_main = pd.read_csv(pred_main_path)

    _save_inverse_metric_barplot(
        df_metrics=df_metrics,
        metric_main_col="R2_main",
        metric_all_col="R2_all",
        ylabel="R-squared ($R^2$)",
        title="Inverse Model Comparison: Main Subset vs All Test ($R^2$)",
        out_path=os.path.join(pic_dir, "inverse_r2_main_vs_all.png"),
    )

    _save_inverse_metric_barplot(
        df_metrics=df_metrics,
        metric_main_col="ARE_main(%)",
        metric_all_col="ARE_all(%)",
        ylabel="Average Relative Error (%)",
        title="Inverse Model Comparison: Main Subset vs All Test (ARE)",
        out_path=os.path.join(pic_dir, "inverse_are_main_vs_all.png"),
    )

    _save_inverse_scatter(
        df_pred=df_pred_main,
        subset_name="Main Subset",
        out_path=os.path.join(pic_dir, "inverse_true_vs_predicted_main.png"),
    )

    _save_inverse_scatter(
        df_pred=df_pred_all,
        subset_name="All Test Samples",
        out_path=os.path.join(pic_dir, "inverse_true_vs_predicted_all.png"),
    )


def main():
    data_dir = "output_data"
    pic_dir = "output_picture"
    os.makedirs(pic_dir, exist_ok=True)

    _save_forward_plots(data_dir, pic_dir)
    _maybe_save_inverse_plots(data_dir, pic_dir)

    print("\n所有绘图完成！图片保存在 output_picture 文件夹中。")


if __name__ == "__main__":
    main()
