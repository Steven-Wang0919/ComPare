# -*- coding: utf-8 -*-
"""
plot_figures.py

为 compare_all.py 输出结果生成论文图像。

输入：
- 优先读取指定 run 目录，例如：
  runs/20260312T120000_compare_all/
    - forward_model_metrics.csv
    - forward_model_predictions.csv
    - inverse_model_metrics.csv
    - inverse_model_predictions_all.csv
    - inverse_model_predictions_main.csv

兼容：
- 若用户显式传入旧目录（如 output_data），也可正常工作

输出：
- 默认保存到 <run_dir>/figures/
- 也可通过 --pic-dir 显式指定

用法：
1) 自动查找最新 compare_all 结果：
   python plot_figures.py

2) 指定某次运行目录：
   python plot_figures.py --run-dir runs/20260312T120000_compare_all

3) 指定输入与输出目录：
   python plot_figures.py --run-dir runs/20260312T120000_compare_all --pic-dir runs/20260312T120000_compare_all/figures
"""

import argparse
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

INVERSE_MODELS = ["inverse_MLP", "inverse_GRNN", "inverse_KAN"]
INVERSE_DISPLAY_MAP = {
    "inverse_MLP": "inverse_MLP",
    "inverse_GRNN": "inverse_GRNN",
    "inverse_KAN": "inverse_KAN",
}
INVERSE_PRED_COL_MAP = {
    "inverse_MLP": "inverse_MLP_pred",
    "inverse_GRNN": "inverse_GRNN_pred",
    "inverse_KAN": "inverse_KAN_pred",
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


def _find_latest_compare_all_run(runs_root="runs"):
    if not os.path.isdir(runs_root):
        return None

    candidates = []
    for name in os.listdir(runs_root):
        full = os.path.join(runs_root, name)
        if not os.path.isdir(full):
            continue
        if name.endswith("_compare_all"):
            metrics_path = os.path.join(full, "forward_model_metrics.csv")
            pred_path = os.path.join(full, "forward_model_predictions.csv")
            if os.path.exists(metrics_path) and os.path.exists(pred_path):
                candidates.append(full)

    if not candidates:
        return None

    candidates.sort()
    return candidates[-1]


def _resolve_run_dir(explicit_run_dir=None):
    if explicit_run_dir:
        if not os.path.isdir(explicit_run_dir):
            raise FileNotFoundError(f"指定的 run_dir 不存在: {explicit_run_dir}")
        return explicit_run_dir

    latest = _find_latest_compare_all_run("runs")
    if latest is not None:
        return latest

    # 兼容旧版 output_data
    if os.path.isdir("output_data"):
        return "output_data"

    raise FileNotFoundError(
        "未找到可用结果目录。请先运行 compare_all.py，"
        "或使用 --run-dir 显式指定 runs/<timestamp>_compare_all。"
    )


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
    true_col = "true_speed_r_min"
    required_cols = [true_col] + [INVERSE_PRED_COL_MAP[m] for m in INVERSE_MODELS]
    _validate_required_columns(df_pred, required_cols, f"inverse predictions ({subset_name})")

    y_true = df_pred[true_col].values
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
        metric_main_col="ARE_all(%)",
        metric_all_col="ARE_main(%)",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=None,
        help="compare_all.py 的输出目录；不传则自动寻找最新 runs/*_compare_all",
    )
    parser.add_argument(
        "--pic-dir",
        default=None,
        help="图片输出目录；默认保存到 <run_dir>/figures",
    )
    args = parser.parse_args()

    data_dir = _resolve_run_dir(args.run_dir)

    if args.pic_dir is not None:
        pic_dir = args.pic_dir
    else:
        pic_dir = os.path.join(data_dir, "figures")

    os.makedirs(pic_dir, exist_ok=True)

    print(f"数据目录: {data_dir}")
    print(f"图片输出目录: {pic_dir}")

    _save_forward_plots(data_dir, pic_dir)
    _maybe_save_inverse_plots(data_dir, pic_dir)

    print("\n所有绘图完成！")


if __name__ == "__main__":
    main()