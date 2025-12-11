# -*- coding: utf-8 -*-
"""
inverse_kan_V1.py

反向 KAN（同构版）：
    - KANLayer 的实现参考 train_kan.py 中的版本（带 input_grid, curve2coeff 等）
    - 网络结构：2 -> hidden_dim(默认8) -> 1
      输入:  [目标质量(g/min), 排肥口开度(mm)]
      输出:  [排肥轴转速(r/min)]

训练与评估：
    - 数据加载: 使用 common_utils.load_data()
      X_raw: [开度(mm), 转速(r/min)]
      y_raw: [质量(g/min)]
    - 构造反向问题:
      输入: [质量, 开度]   -> X_inv
      输出: 转速          -> y_inv
    - 归一化:
      仅用 train+val 的样本统计 x_min/x_max 和 y_min/y_max，做 0–1 归一化
    - 训练:
      使用 AdamW + ExponentialLR
    - 评估:
      在 test 集上，仅筛选“实际开度 == 策略开度”的样本
      计算 R² 和平均相对误差 MRS，并打印典型样本（逻辑与 inverse_kan.py 一致）
"""

import numpy as np
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common_utils import load_data, get_train_val_test_indices, mean_relative_error


# =========================
# 1. 策略配置（与原 inverse_kan.py 保持一致）
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


# =========================
# 2. KAN 基础层（同构版，参考 train_kan.py）
# =========================
class KANLayer(nn.Module):
    # 默认 grid_size = 10
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

        # 输入网格：每个输入维度一条 1D 网格
        self.input_grid = torch.einsum(
            "i,j->ij",
            torch.ones(in_features),
            torch.linspace(grid_range[0], grid_range[1], grid_size + 1),
        )
        # 不参与训练
        self.input_grid = nn.Parameter(self.input_grid, requires_grad=False)

        # 线性（基函数）权重
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # 样条部分权重
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        # 线性部分初始化
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)

        # 样条部分初始化
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            # noise: [grid+1, in_features, out_features]
            coeff = self.curve2coeff(self.input_grid, noise)
            # coeff: [out_features, in_features, grid + spline_order]
            self.spline_weight.data.copy_(
                (self.scale_spline if self.scale_spline is not None else 1.0) * coeff
            )

    def b_splines(self, x: torch.Tensor):
        """
        x: [batch, in_features]，假定范围大致在 grid_range 内
        返回样条基底值: [batch, in_features, grid_size + spline_order]
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.input_grid  # [in_features, grid+1]
        h = (grid[:, -1:] - grid[:, 0:1]) / self.grid_size
        device = grid.device

        # 左右 padding
        arange_left = torch.arange(
            self.spline_order, 0, -1, device=device, dtype=grid.dtype
        ).unsqueeze(0)
        left_pad = grid[:, 0:1] - arange_left * h

        arange_right = torch.arange(
            1, self.spline_order + 1, device=device, dtype=grid.dtype
        ).unsqueeze(0)
        right_pad = grid[:, -1:] + arange_right * h

        grid = torch.cat([left_pad, grid, right_pad], dim=1)  # [in_features, grid+1+2*order]

        x = x.unsqueeze(-1)          # [B, in_features, 1]
        grid = grid.unsqueeze(0)     # [1, in_features, G]

        # 0阶 B-spline：区间指示函数
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)

        # 递推构造高阶 B-spline
        for k in range(1, self.spline_order + 1):
            denom1 = grid[:, :, k:-1] - grid[:, :, : -(k + 1)]
            denom2 = grid[:, :, k + 1 :] - grid[:, :, 1:-k]
            term1 = (x - grid[:, :, : -(k + 1)]) / denom1 * bases[:, :, :-1]
            term2 = (grid[:, :, k + 1 :] - x) / denom2 * bases[:, :, 1:]
            bases = term1 + term2

        return bases.contiguous()  # [B, in_features, grid_size + spline_order]

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        给定一条曲线 (x, y)，求样条系数
        x: [in_features, grid+1]
        y: [grid+1, in_features, out_features]
        """
        A = self.b_splines(x.transpose(0, 1)).transpose(0, 1)  # [in_features, grid+1, grid+order]
        B = y.transpose(0, 1)                                  # [in_features, grid+1, out_features]
        solution = torch.linalg.lstsq(A, B).solution           # [in_features, grid+order, out_features]
        result = solution.permute(2, 0, 1)                     # [out_features, in_features, grid+order]
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        # 线性部分
        base_out = F.linear(self.base_activation(x), self.base_weight)
        # 样条部分
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out


# =========================
# 3. 整体“反向 KAN”网络
# =========================
class InverseFertilizerKAN(nn.Module):
    """
    反向 KAN：
        输入:  [质量, 开度]  (经 0-1 归一化)
        输出:  [转速]        (经 0-1 归一化)
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
# 4. 训练 & 策略一致性评估
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- A. 数据加载 ---
    # X_raw: [开度(mm), 转速(r/min)]
    # y_raw: [质量(g/min)]
    X_raw, y_raw = load_data()
    n_samples = len(y_raw)

    train_idx, val_idx, test_idx = get_train_val_test_indices(n_samples)

    # 构造反向问题：输入 [质量, 开度]，输出 [转速]
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)  # (N, 2)
    y_inv_all = X_raw[:, 1]                             # (N,)

    # 训练集（物理规律学习使用 train+val）
    train_full_idx = np.concatenate([train_idx, val_idx])

    X_train_full_raw = X_inv_all[train_full_idx]    # [mass, opening]
    y_train_full_raw = y_inv_all[train_full_idx]    # speed

    # --- B. 归一化（仅用 train+val 统计） ---
    x_min = X_train_full_raw.min(axis=0, keepdims=True)
    x_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = y_train_full_raw.min()
    y_max = y_train_full_raw.max()

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + 1e-8)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + 1e-8)

    X_train_norm = norm_x(X_train_full_raw)
    y_train_norm = norm_y(y_train_full_raw)

    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).view(-1, 1).to(device)

    # --- C. 训练反向 KAN（同构版） ---
    print(">>> 正在训练同构版反向 KAN 模型...")

    model = InverseFertilizerKAN(input_dim=2, hidden_dim=8, output_dim=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()

    epochs = 600
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss = {loss.item():.6f}")

    # --- D. 在测试集上做“策略一致性”评估（与 inverse_kan.py 一样） ---
    print("\n>>> 开始验证 (仅筛选符合策略的样本)...")
    model.eval()

    test_mass = y_raw[test_idx]        # 目标质量
    test_opening = X_raw[test_idx, 0]  # 实际开度
    test_speed_true = X_raw[test_idx, 1]

    valid_indices = []

    for i in range(len(test_mass)):
        m = float(test_mass[i])          # 目标质量
        real_op = float(test_opening[i]) # 实际开度

        strat_op = select_optimal_opening(m)

        # 只有 "实际开度 == 策略开度" 的样本，才纳入评估
        if np.isclose(real_op, strat_op, atol=0.1):
            valid_indices.append(i)

    if len(valid_indices) == 0:
        print("测试集中未找到符合当前策略 (20/35/50mm) 的样本。")
    else:
        f_mass = test_mass[valid_indices]
        f_op = test_opening[valid_indices]
        f_spd_true = test_speed_true[valid_indices]

        # 组合输入并归一化
        input_vec = np.stack([f_mass, f_op], axis=1)
        input_norm = norm_x(input_vec)
        input_t = torch.tensor(input_norm, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_norm = model(input_t).cpu().numpy().reshape(-1)

        # 反归一化到真实转速
        pred_spd = pred_norm * (y_max - y_min + 1e-8) + y_min

        # 计算指标
        r2 = r2_score(f_spd_true, pred_spd)
        mrs = mean_relative_error(f_spd_true, pred_spd)

        print(f"\n===== 同构版反向 KAN - 策略一致性样本评估 (共 {len(valid_indices)} 个) =====")
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
