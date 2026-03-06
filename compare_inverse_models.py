# -*- coding: utf-8 -*-
"""
compare_inverse_models.py

对比三个“反向模型”：
    - GRNN（inverse_grnn.InverseGRNN）
    - MLP  （inverse_mlp.train_and_eval_inverse_mlp）
    - KAN  （inverse_kan_V2.InverseFertilizerKAN）

统一：
    - 数据：common_utils.load_data()
    - 划分：common_utils.get_train_val_test_indices()
    - 任务：输入 [目标质量, 开度]，输出 [转速]
    - 归一化：输入/输出都使用 train+val 统计做 0-1 归一化
    - 策略样本：只评估 “实际开度 == 策略开度” 的 test 样本
    - 指标：R²、平均相对误差 (MRS)

可视化：
    - ComPare_Pic/metrics_bar.png
    - ComPare_Pic/scatter_true_vs_pred.png
"""

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, mean_relative_error
from inverse_grnn import InverseGRNN
from inverse_mlp import train_and_eval_inverse_mlp
from inverse_kan_V2 import InverseFertilizerKAN


matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# =============== 0. 固定随机种子，增强可复现性 ===============
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


# =============== 公共策略（保持与各文件一致） ===============
THRESHOLD_LOW_MID = 2800.0   # 20mm -> 35mm
THRESHOLD_MID_HIGH = 4800.0  # 35mm -> 50mm


def select_optimal_opening(target_mass: float) -> float:
    """根据目标排肥量，自动决定最佳开度（与各脚本保持一致）"""
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    elif target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    else:
        return 50.0


# =============== 1. GRNN：带 val R² 选 σ 的训练 & 评估 ===============
def run_grnn(data_path="data/dataset.xlsx"):
    print("\n=== 训练 & 评估 反向 GRNN ===")

    # --- A. 加载数据并构造“反向问题” ---
    X_raw, y_raw = load_data(data_path)  # X_raw: [开度, 转速], y_raw: [质量]
    n_samples = len(y_raw)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(n_samples)

    # 输入 [质量, 开度]，输出 [转速]
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    # --- B. 使用 train+val 统计归一化参数（输入+输出） ---
    train_full_idx = np.concatenate([idx_tr, idx_val])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    x_min = X_train_full_raw.min(axis=0, keepdims=True)
    x_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_full_raw.min())
    y_max = float(y_train_full_raw.max())

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + 1e-8)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + 1e-8)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + 1e-8) + y_min

    # --- C. 在 train 上训练，在 val 上用 R² 选 σ ---
    sigma_grid = np.linspace(0.1, 4.0, 40)

    X_train_raw = X_inv_all[idx_tr]
    y_train_raw = y_inv_all[idx_tr]

    X_val_raw = X_inv_all[idx_val]
    y_val_raw = y_inv_all[idx_val]

    X_train_norm = norm_x(X_train_raw)
    y_train_norm = norm_y(y_train_raw)
    X_val_norm = norm_x(X_val_raw)

    best_sigma = None
    best_r2_val = -np.inf

    print(">>> 为 GRNN 选择最优 σ（基于 val R²）...")
    for s in sigma_grid:
        model_tmp = InverseGRNN(sigma=s)
        model_tmp.fit(X_train_norm, y_train_norm)

        # 在 val 集上预测归一化转速，再反归一化后评估
        y_val_pred_norm = model_tmp.predict(X_val_norm)
        y_val_pred = denorm_y(y_val_pred_norm)

        r2_val = r2_score(y_val_raw, y_val_pred)
        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_sigma = s

    print(f"GRNN 最优 σ = {best_sigma:.4f}, val R² = {best_r2_val:.6f}")

    # --- D. 用 train+val 训练最终 GRNN ---
    X_train_full_norm = norm_x(X_train_full_raw)
    y_train_full_norm = norm_y(y_train_full_raw)

    grnn_final = InverseGRNN(sigma=best_sigma)
    grnn_final.fit(X_train_full_norm, y_train_full_norm)

    # --- E. 在 test 上做“策略一致性”评估 ---
    print(">>> GRNN 测试集策略一致性评估 ...")

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    valid_indices = []
    for i in range(len(test_mass)):
        m = float(test_mass[i])
        real_op = float(test_opening[i])
        strat_op = select_optimal_opening(m)
        if np.isclose(real_op, strat_op, atol=0.1):
            valid_indices.append(i)

    if len(valid_indices) == 0:
        print("GRNN: 测试集中未找到符合当前策略的样本。")
        return None

    f_mass = test_mass[valid_indices]
    f_op = test_opening[valid_indices]
    f_spd_true = test_speed_true[valid_indices]

    input_vec = np.stack([f_mass, f_op], axis=1)
    input_norm = norm_x(input_vec)

    pred_spd_norm = grnn_final.predict(input_norm)
    pred_spd = denorm_y(pred_spd_norm)

    r2 = r2_score(f_spd_true, pred_spd)
    mrs = mean_relative_error(f_spd_true, pred_spd)

    print(f"GRNN: R²={r2:.4f}, MRS={mrs:.4f}% (样本数={len(valid_indices)})")

    return {
        "name": "GRNN",
        "r2": r2,
        "mrs": mrs,
        "y_true": np.asarray(f_spd_true),
        "y_pred": np.asarray(pred_spd),
        "mass": np.asarray(f_mass),
        "residual": np.asarray(pred_spd) - np.asarray(f_spd_true),
    }


# =============== 2. MLP：直接调用你已有的训练函数 ===============
def run_mlp(data_path="data/dataset.xlsx"):
    print("\n=== 训练 & 评估 反向 MLP ===")

    # 要求 inverse_mlp.train_and_eval_inverse_mlp 返回:
    # r2, mrs, y_true_valid, y_pred_valid, mass_valid
    res = train_and_eval_inverse_mlp(data_path=data_path)
    if res is None:
        print("MLP: 无有效测试样本。")
        return None

    required_keys = ["r2", "mrs", "y_true_valid", "y_pred_valid", "mass_valid"]
    for k in required_keys:
        if k not in res:
            raise KeyError(f"inverse_mlp 返回结果缺少必要字段: {k}")

    return {
        "name": "MLP",
        "r2": res["r2"],
        "mrs": res["mrs"],
        "y_true": np.asarray(res["y_true_valid"]),
        "y_pred": np.asarray(res["y_pred_valid"]),
        "mass": np.asarray(res["mass_valid"]),
        "residual": np.asarray(res["y_pred_valid"]) - np.asarray(res["y_true_valid"]),
    }


# =============== 3. KAN V2：基于 inverse_kan_V2 的逻辑封装成函数 ===============
def run_kan_v2(data_path="data/dataset.xlsx"):
    print("\n=== 训练 & 评估 反向 KAN ===")

    X_raw, y_raw = load_data(data_path)
    n_samples = len(y_raw)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(n_samples)

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]
    y_test_raw = y_inv_all[idx_te]

    # 归一化参数（train+val）
    train_full_idx = np.concatenate([idx_tr, idx_val])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    x_min = X_train_full_raw.min(axis=0, keepdims=True)
    x_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_full_raw.min())
    y_max = float(y_train_full_raw.max())

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + 1e-8)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + 1e-8)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + 1e-8) + y_min

    X_train_norm = norm_x(X_train_raw)
    X_val_norm = norm_x(X_val_raw)
    X_test_norm = norm_x(X_test_raw)

    y_train_norm = norm_y(y_train_raw)
    y_val_norm = norm_y(y_val_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("KAN 使用设备:", device)

    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    # --- 超参搜索 ---
    hidden_dim_candidates = [4, 8, 16]
    lr_candidates = [0.01, 0.005]
    weight_decay_candidates = [1e-4, 1e-5]
    search_epochs = 300
    gamma = 0.99

    criterion = nn.MSELoss()
    best_r2_val = -np.inf
    best_cfg = None

    print(">>> 开始 KAN 超参搜索 ...")
    for hidden_dim in hidden_dim_candidates:
        for lr in lr_candidates:
            for wd in weight_decay_candidates:
                set_seed(42)

                model = InverseFertilizerKAN(
                    input_dim=2, hidden_dim=hidden_dim, output_dim=1
                ).to(device)

                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

                for epoch in range(search_epochs):
                    model.train()
                    optimizer.zero_grad()
                    pred_train = model(X_train_t)
                    loss = criterion(pred_train, y_train_t)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                model.eval()
                with torch.no_grad():
                    pred_val_norm = model(X_val_t).cpu().numpy().reshape(-1)
                y_val_pred = denorm_y(pred_val_norm)

                r2_val = r2_score(y_val_raw, y_val_pred)
                print(
                    f"  [search] hidden_dim={hidden_dim}, lr={lr}, wd={wd}, "
                    f"val R²={r2_val:.6f}"
                )

                if r2_val > best_r2_val:
                    best_r2_val = r2_val
                    best_cfg = (hidden_dim, lr, wd)

    if best_cfg is None:
        raise RuntimeError("KAN 超参搜索失败：best_cfg 为空")

    hidden_best, lr_best, wd_best = best_cfg
    print(
        f"KAN 最优超参: hidden_dim={hidden_best}, lr={lr_best}, "
        f"weight_decay={wd_best}, val R²={best_r2_val:.6f}"
    )

    # --- 用 train+val 训练最终模型 ---
    X_train_val_norm = norm_x(np.vstack([X_train_raw, X_val_raw]))
    y_train_val_norm = norm_y(np.hstack([y_train_raw, y_val_raw]))

    X_train_val_t = torch.tensor(X_train_val_norm, dtype=torch.float32).to(device)
    y_train_val_t = torch.tensor(y_train_val_norm, dtype=torch.float32).view(-1, 1).to(device)

    set_seed(42)
    model_final = InverseFertilizerKAN(
        input_dim=2, hidden_dim=hidden_best, output_dim=1
    ).to(device)

    optimizer = optim.AdamW(model_final.parameters(), lr=lr_best, weight_decay=wd_best)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.MSELoss()

    epochs_final = 600
    print(">>> 开始训练最终 KAN 模型 ...")
    for epoch in range(epochs_final):
        model_final.train()
        optimizer.zero_grad()
        pred = model_final(X_train_val_t)
        loss = criterion(pred, y_train_val_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{epochs_final}, Train Loss={loss.item():.6f}")

    # --- 测试集策略一致性评估 ---
    print(">>> KAN 测试集策略一致性评估 ...")

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    valid_indices = []
    for i in range(len(test_mass)):
        m = float(test_mass[i])
        real_op = float(test_opening[i])
        strat_op = select_optimal_opening(m)
        if np.isclose(real_op, strat_op, atol=0.1):
            valid_indices.append(i)

    if len(valid_indices) == 0:
        print("KAN: 测试集中未找到符合当前策略的样本。")
        return None

    f_mass = test_mass[valid_indices]
    f_op = test_opening[valid_indices]
    f_spd_true = test_speed_true[valid_indices]

    input_vec = np.stack([f_mass, f_op], axis=1)
    input_norm = norm_x(input_vec)
    input_t = torch.tensor(input_norm, dtype=torch.float32).to(device)

    model_final.eval()
    with torch.no_grad():
        pred_norm = model_final(input_t).cpu().numpy().reshape(-1)
    pred_spd = denorm_y(pred_norm)

    r2 = r2_score(f_spd_true, pred_spd)
    mrs = mean_relative_error(f_spd_true, pred_spd)

    print(f"KAN: R²={r2:.4f}, MRS={mrs:.4f}% (样本数={len(valid_indices)})")

    return {
        "name": "KAN",
        "r2": r2,
        "mrs": mrs,
        "y_true": np.asarray(f_spd_true),
        "y_pred": np.asarray(pred_spd),
        "mass": np.asarray(f_mass),
        "residual": np.asarray(pred_spd) - np.asarray(f_spd_true),
    }


# =============== 4. 结果一致性校验 ===============
def validate_results_alignment(results, atol=1e-8):
    """
    强校验：三模型必须在同一批测试样本上比较
    """
    if len(results) <= 1:
        return

    ref = results[0]
    ref_true = np.asarray(ref["y_true"]).reshape(-1)
    ref_mass = np.asarray(ref["mass"]).reshape(-1)

    for r in results[1:]:
        cur_true = np.asarray(r["y_true"]).reshape(-1)
        cur_mass = np.asarray(r["mass"]).reshape(-1)

        if len(cur_true) != len(ref_true):
            raise ValueError(
                f"模型 {r['name']} 与 {ref['name']} 的有效测试样本数不一致: "
                f"{len(cur_true)} vs {len(ref_true)}"
            )

        if not np.allclose(cur_true, ref_true, atol=atol, rtol=0):
            raise ValueError(
                f"模型 {r['name']} 与 {ref['name']} 的 y_true 不一致，不能直接公平对比。"
            )

        if not np.allclose(cur_mass, ref_mass, atol=atol, rtol=0):
            raise ValueError(
                f"模型 {r['name']} 与 {ref['name']} 的 mass 不一致，说明筛样口径不同。"
            )


# =============== 5. 可视化函数 ===============
def plot_metrics(results, out_dir):
    models = [r["name"] for r in results]
    r2s = [r["r2"] for r in results]
    mrss = [r["mrs"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ----- R^2 -----
    axes[0].bar(models, r2s)
    axes[0].set_title(r"$R^2$ 对比", fontsize=14)
    axes[0].set_ylabel(r"$R^2$", fontsize=12)

    r2_min, r2_max = 0, max(r2s)
    axes[0].set_ylim(r2_min, r2_max * 1.10 if r2_max > 0 else 1.0)
    dy = (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]) * 0.03

    for x, v in zip(models, r2s):
        axes[0].text(x, v + dy, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    axes[0].tick_params(axis="both", labelsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # ----- MRS -----
    axes[1].bar(models, mrss)
    axes[1].set_title("平均相对误差 MRS 对比", fontsize=14)
    axes[1].set_ylabel("MRS (%)", fontsize=12)

    mrs_min, mrs_max = 0, max(mrss)
    axes[1].set_ylim(mrs_min, mrs_max * 1.20 if mrs_max > 0 else 1.0)
    dy2 = (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]) * 0.03

    for x, v in zip(models, mrss):
        axes[1].text(x, v + dy2, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    axes[1].tick_params(axis="both", labelsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "metrics_bar.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存指标对比图: {save_path}")


def plot_scatter(results, out_dir):
    """三模型真实转速 vs 预测转速散点图（论文级）"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharex=True, sharey=True)

    if n == 1:
        axes = [axes]

    all_true = np.concatenate([r["y_true"] for r in results])
    all_pred = np.concatenate([r["y_pred"] for r in results])
    vmin = min(all_true.min(), all_pred.min())
    vmax = max(all_true.max(), all_pred.max())

    for ax, r in zip(axes, results):
        y_true = np.asarray(r["y_true"])
        y_pred = np.asarray(r["y_pred"])

        ax.scatter(y_true, y_pred, s=30, alpha=0.8, edgecolors="none")
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1)

        ax.set_title(
            f"{r['name']}  ($R^2$={r['r2']:.4f}, MRS={r['mrs']:.2f}%)",
            fontsize=13
        )
        ax.set_xlabel("真实转速 (r/min)", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)

    axes[0].set_ylabel("预测转速 (r/min)", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "scatter_true_vs_pred.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存散点对比图: {save_path}")


# =============== 6. 主函数 ===============
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

    validate_results_alignment(results)

    print("\n===== 三模型指标汇总 =====")
    for r in results:
        print(f"{r['name']:<8} | R²={r['r2']:.4f} | MRS={r['mrs']:.4f}% | N={len(r['y_true'])}")

    plot_metrics(results, out_dir)
    plot_scatter(results, out_dir)


if __name__ == "__main__":
    main()