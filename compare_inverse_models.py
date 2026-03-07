# -*- coding: utf-8 -*-
"""
compare_inverse_models.py

对比三个“反向模型”：
    - GRNN
    - MLP
    - KAN

统一口径（与论文一致）：
    - 数据：common_utils.load_data()
    - 划分：common_utils.get_train_val_test_indices()
    - 任务：输入 [目标质量, 开度]，输出 [转速]
    - 归一化：输入/输出都使用 train+val 统计做 0-1 归一化
    - 主评估对象：策略一致子集（实际开度 == 策略开度）
    - 补充评估对象：全测试集（仅作透明展示，不作为主结论）
    - 指标：R²、平均相对误差 (ARE)

可视化（基于主结果）：
    - ComPare_Pic/metrics_bar.png
    - ComPare_Pic/scatter_true_vs_pred.png
"""

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from inverse_grnn import train_and_eval_inverse_grnn
from inverse_mlp import train_and_eval_inverse_mlp
from inverse_kan import train_and_eval_inverse_kan_v2


matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def standardize_result(model_name, res):
    if res is None:
        return None

    required_keys = [
        "r2_main", "are_main", "n_main",
        "r2_all", "are_all", "n_all",
        "y_true_main", "y_pred_main", "mass_main",
        "y_true_all", "y_pred_all", "mass_all",
    ]
    for k in required_keys:
        if k not in res:
            raise KeyError(f"{model_name} 返回结果缺少必要字段: {k}")

    return {
        "name": model_name,

        "r2_main": float(res["r2_main"]) if not np.isnan(res["r2_main"]) else np.nan,
        "are_main": float(res["are_main"]) if not np.isnan(res["are_main"]) else np.nan,
        "n_main": int(res["n_main"]),
        "y_true_main": np.asarray(res["y_true_main"]),
        "y_pred_main": np.asarray(res["y_pred_main"]),
        "mass_main": np.asarray(res["mass_main"]),
        "residual_main": np.asarray(res["y_pred_main"]) - np.asarray(res["y_true_main"]),

        "r2_all": float(res["r2_all"]),
        "are_all": float(res["are_all"]),
        "n_all": int(res["n_all"]),
        "y_true_all": np.asarray(res["y_true_all"]),
        "y_pred_all": np.asarray(res["y_pred_all"]),
        "mass_all": np.asarray(res["mass_all"]),
        "residual_all": np.asarray(res["y_pred_all"]) - np.asarray(res["y_true_all"]),
    }


def run_grnn(data_path="data/dataset.xlsx"):
    print("\n=== 训练 & 评估 反向 GRNN ===")
    res = train_and_eval_inverse_grnn(data_path=data_path)
    return standardize_result("GRNN", res)


def run_mlp(data_path="data/dataset.xlsx"):
    print("\n=== 训练 & 评估 反向 MLP ===")
    res = train_and_eval_inverse_mlp(data_path=data_path)
    return standardize_result("MLP", res)


def run_kan_v2(data_path="data/dataset.xlsx"):
    print("\n=== 训练 & 评估 反向 KAN ===")
    res = train_and_eval_inverse_kan_v2(data_path=data_path)
    return standardize_result("KAN", res)


def validate_results_alignment(results, atol=1e-8):
    if len(results) <= 1:
        return

    ref = results[0]
    ref_true = np.asarray(ref["y_true_main"]).reshape(-1)
    ref_mass = np.asarray(ref["mass_main"]).reshape(-1)

    for r in results[1:]:
        cur_true = np.asarray(r["y_true_main"]).reshape(-1)
        cur_mass = np.asarray(r["mass_main"]).reshape(-1)

        if len(cur_true) != len(ref_true):
            raise ValueError(
                f"模型 {r['name']} 与 {ref['name']} 的主评估样本数不一致: "
                f"{len(cur_true)} vs {len(ref_true)}"
            )

        if not np.allclose(cur_true, ref_true, atol=atol, rtol=0):
            raise ValueError(
                f"模型 {r['name']} 与 {ref['name']} 的主评估 y_true 不一致，不能直接公平对比。"
            )

        if not np.allclose(cur_mass, ref_mass, atol=atol, rtol=0):
            raise ValueError(
                f"模型 {r['name']} 与 {ref['name']} 的主评估 mass 不一致，说明筛样口径不同。"
            )


def plot_metrics(results, out_dir):
    models = [r["name"] for r in results]
    r2s = [r["r2_main"] for r in results]
    ares = [r["are_main"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(models, r2s)
    axes[0].set_title(r"主结果：$R^2$ 对比", fontsize=14)
    axes[0].set_ylabel(r"$R^2$", fontsize=12)

    r2_min, r2_max = 0, max(r2s) if len(r2s) > 0 else 1.0
    axes[0].set_ylim(r2_min, r2_max * 1.10 if r2_max > 0 else 1.0)
    dy = (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]) * 0.03

    for x, v in zip(models, r2s):
        axes[0].text(x, v + dy, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    axes[0].tick_params(axis="both", labelsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].bar(models, ares)
    axes[1].set_title("主结果：平均相对误差 ARE 对比", fontsize=14)
    axes[1].set_ylabel("ARE (%)", fontsize=12)

    are_min, are_max = 0, max(ares) if len(ares) > 0 else 1.0
    axes[1].set_ylim(are_min, are_max * 1.20 if are_max > 0 else 1.0)
    dy2 = (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]) * 0.03

    for x, v in zip(models, ares):
        axes[1].text(x, v + dy2, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    axes[1].tick_params(axis="both", labelsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "metrics_bar.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存主结果指标对比图: {save_path}")


def plot_scatter(results, out_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharex=True, sharey=True)

    if n == 1:
        axes = [axes]

    all_true = np.concatenate([r["y_true_main"] for r in results])
    all_pred = np.concatenate([r["y_pred_main"] for r in results])
    vmin = min(all_true.min(), all_pred.min())
    vmax = max(all_true.max(), all_pred.max())

    for ax, r in zip(axes, results):
        y_true = np.asarray(r["y_true_main"])
        y_pred = np.asarray(r["y_pred_main"])

        ax.scatter(y_true, y_pred, s=30, alpha=0.8, edgecolors="none")
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1)

        ax.set_title(
            f"{r['name']}  ($R^2$={r['r2_main']:.4f}, ARE={r['are_main']:.2f}%)",
            fontsize=13
        )
        ax.set_xlabel("真实转速 (r/min)", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)

    axes[0].set_ylabel("预测转速 (r/min)", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "scatter_true_vs_pred.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存主结果散点对比图: {save_path}")


def print_summary(results):
    print("\n===== 三模型主结果汇总（论文主口径：策略一致子集） =====")
    for r in results:
        print(
            f"{r['name']:<8} | "
            f"R²={r['r2_main']:.4f} | "
            f"ARE={r['are_main']:.4f}% | "
            f"N={r['n_main']}"
        )

    print("\n===== 三模型补充结果汇总（全测试集，仅透明展示） =====")
    for r in results:
        print(
            f"{r['name']:<8} | "
            f"R²={r['r2_all']:.4f} | "
            f"ARE={r['are_all']:.4f}% | "
            f"N={r['n_all']}"
        )


def main():
    set_seed(42)

    out_dir = os.path.join(os.path.dirname(__file__), "ComPare_Pic")
    os.makedirs(out_dir, exist_ok=True)

    res_grnn = run_grnn()
    res_mlp = run_mlp()
    res_kan = run_kan_v2()

    results = [r for r in [res_grnn, res_mlp, res_kan] if r is not None]

    if len(results) == 0:
        print("没有任何模型产生有效结果，无法绘图。")
        return

    results = [r for r in results if r["n_main"] > 0]
    if len(results) == 0:
        print("所有模型的主评估样本数都为 0，无法进行公平对比。")
        return

    validate_results_alignment(results)
    print_summary(results)
    plot_metrics(results, out_dir)
    plot_scatter(results, out_dir)


if __name__ == "__main__":
    main()