# -*- coding: utf-8 -*-
"""
inverse_kan_V2.py

带超参搜索的反向 KAN（同构版）：

    输入:  [目标质量(g/min), 排肥口开度(mm)]
    输出:  [排肥轴转速(r/min)]

特性:
    - KANLayer 结构与 train_kan.py 中一致（input_grid + curve2coeff + B-spline）
    - 输入、输出全部进行 0–1 归一化
      * 归一化范围使用 train+val (train_full_idx)，与其他反向模型保持一致
    - 在 train 上训练，在 val 上通过 R² 搜索超参（hidden_dim, lr, weight_decay）
    - 用最优超参在 train+val 上重新训练最终模型
    - 在 test 集上，仅对“实际开度 == 策略开度”的样本进行 R² / MRS 评估
"""

import random
import numpy as np
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common_utils import load_data, get_train_val_test_indices, mean_relative_error


# =========================
# 1. 策略配置（与其它 inverse_* 保持一致）
# =========================
THRESHOLD_LOW_MID = 2800.0   # 20mm -> 35mm 切换点
THRESHOLD_MID_HIGH = 4800.0  # 35mm -> 50mm 切换点


def select_optimal_opening(target_mass: float) -> float:
    """
    根据目标排肥量，自动决定最佳开度
    """
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    elif target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    else:
        return 50.0


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


# =========================
# 2. KAN 基础层（同构版，参考 train_kan.py）
# =========================
class KANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=10,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_range=(-1.0, 1.0),
    ):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()

        self.input_grid = torch.einsum(
            "i,j->ij",
            torch.ones(in_features),
            torch.linspace(grid_range[0], grid_range[1], grid_size + 1),
        )
        self.input_grid = nn.Parameter(self.input_grid, requires_grad=False)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            coeff = self.curve2coeff(self.input_grid, noise)
            self.spline_weight.data.copy_(
                (self.scale_spline if self.scale_spline is not None else 1.0) * coeff
            )

    def b_splines(self, x: torch.Tensor):
        """
        x: [batch, in_features]
        返回: [batch, in_features, grid_size + spline_order]
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.input_grid
        h = (grid[:, -1:] - grid[:, 0:1]) / self.grid_size
        device = grid.device

        arange_left = torch.arange(
            self.spline_order, 0, -1, device=device, dtype=grid.dtype
        ).unsqueeze(0)
        left_pad = grid[:, 0:1] - arange_left * h

        arange_right = torch.arange(
            1, self.spline_order + 1, device=device, dtype=grid.dtype
        ).unsqueeze(0)
        right_pad = grid[:, -1:] + arange_right * h

        grid = torch.cat([left_pad, grid, right_pad], dim=1)

        x = x.unsqueeze(-1)
        grid = grid.unsqueeze(0)

        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            denom1 = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            denom2 = grid[:, :, k + 1:] - grid[:, :, 1:-k]
            term1 = (x - grid[:, :, :-(k + 1)]) / (denom1 + 1e-12) * bases[:, :, :-1]
            term2 = (grid[:, :, k + 1:] - x) / (denom2 + 1e-12) * bases[:, :, 1:]
            bases = term1 + term2

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: [in_features, grid+1]
        y: [grid+1, in_features, out_features]
        """
        A = self.b_splines(x.transpose(0, 1)).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out


# =========================
# 3. 反向 KAN 网络
# =========================
class InverseFertilizerKAN(nn.Module):
    """
    反向 KAN：
        输入:  [质量_norm, 开度_norm]
        输出:  [转速_norm]
    """
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        super(InverseFertilizerKAN, self).__init__()
        self.kan1 = KANLayer(input_dim, hidden_dim, grid_size=10)
        self.kan2 = KANLayer(hidden_dim, output_dim, grid_size=10)

    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        return x


# =========================
# 4. 训练 + 超参搜索 + 策略一致性评估
# =========================
def train_and_eval_inverse_kan_v2(data_path="data/dataset.xlsx"):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- A. 数据加载 ---
    # X_raw: [开度(mm), 转速(r/min)]
    # y_raw: [质量(g/min)]
    X_raw, y_raw = load_data(data_path)
    n_samples = len(y_raw)

    idx_tr, idx_val, idx_te = get_train_val_test_indices(n_samples)

    # 反向问题：输入 [质量, 开度]，输出 [转速]
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]
    y_test_raw = y_inv_all[idx_te]

    _ = X_test_raw, y_test_raw

    # --- B. 使用 train+val 统计归一化参数 ---
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
    y_train_norm = norm_y(y_train_raw)

    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    # --- C. 超参数搜索 ---
    hidden_dim_candidates = [4, 8, 16]
    lr_candidates = [0.01, 0.005]
    weight_decay_candidates = [1e-4, 1e-5]
    search_epochs = 300
    gamma = 0.99

    criterion = nn.MSELoss()
    best_r2_val = -np.inf
    best_cfg = None

    print("\n>>> 开始反向 KAN 超参搜索 ...")

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
        raise RuntimeError("反向 KAN 超参搜索失败：best_cfg 为空")

    print(
        f"\n>>> 反向 KAN 最优超参: "
        f"hidden_dim={best_cfg[0]}, lr={best_cfg[1]}, weight_decay={best_cfg[2]}, "
        f"val R²={best_r2_val:.6f}"
    )

    # --- D. 用 train+val 训练最终模型 ---
    hidden_best, lr_best, wd_best = best_cfg

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
    print("\n>>> 开始训练最终反向 KAN 模型 ...")
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

    # --- E. 在 test 集上做“策略一致性”评估 ---
    print("\n>>> 开始在测试集上验证 (仅筛选符合策略的样本)...")

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
        print("测试集中未找到符合当前策略 (20/35/50mm) 的样本。")
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

    print(
        f"\n===== 反向 KAN V2 (带超参搜索) - "
        f"策略一致性样本评估 (共 {len(valid_indices)} 个) ====="
    )
    print(f"R² Score: {r2:.4f}")
    print(f"平均相对误差 (MRS): {mrs:.4f}%")

    print("\n[典型样本详情]")
    print(
        f"{'目标质量':<10} | {'策略开度':<10} | {'实际开度':<10} | "
        f"{'预测转速':<10} | {'实际转速':<10} | {'误差(%)'}"
    )
    for k in range(min(5, len(f_mass))):
        err = abs(pred_spd[k] - f_spd_true[k]) / (f_spd_true[k] + 1e-8) * 100
        print(
            f"{f_mass[k]:<10.1f} | "
            f"{f_op[k]:<10.1f} | "
            f"{f_op[k]:<10.1f} | "
            f"{pred_spd[k]:<10.2f} | "
            f"{f_spd_true[k]:<10.2f} | "
            f"{err:.2f}%"
        )

    return {
        "r2": float(r2),
        "mrs": float(mrs),
        "best_hidden_dim": int(hidden_best),
        "best_lr": float(lr_best),
        "best_weight_decay": float(wd_best),
        "n_valid": int(len(valid_indices)),
        "y_true_valid": np.asarray(f_spd_true),
        "y_pred_valid": np.asarray(pred_spd),
        "mass_valid": np.asarray(f_mass),
    }


if __name__ == "__main__":
    train_and_eval_inverse_kan_v2()