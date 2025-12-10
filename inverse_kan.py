# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from common_utils import load_data, get_train_val_test_indices, mean_relative_error

# =========================
# 1. 策略配置
# =========================
THRESHOLD_LOW_MID = 2800.0  # 20mm -> 35mm 切换点
THRESHOLD_MID_HIGH = 4800.0  # 35mm -> 50mm 切换点


def select_optimal_opening(target_mass):
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
# 2. KAN 模型定义
# =========================
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3, scale_base=1.0, scale_spline=1.0):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = 2.0 / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h - 1.0).expand(in_features,
                                                                                            -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        nn.init.kaiming_uniform_(self.spline_weight, a=np.sqrt(5) * scale_spline)
        self.base_activation = nn.SiLU()

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def forward(self, x):
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        spline_output = nn.functional.linear(self.b_splines(x).view(x.size(0), -1),
                                             self.spline_weight.view(self.out_features, -1))
        return base_output + spline_output


class InverseKAN(nn.Module):
    def __init__(self):
        super(InverseKAN, self).__init__()
        # 输入: [目标质量, 开度] -> 输出: [转速]
        self.layer1 = KANLayer(2, 5)
        self.layer2 = KANLayer(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# =========================
# 3. 训练与特定样本验证
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # --- A. 数据加载 ---
    # X_raw=[开度, 转速], y_raw=[质量]
    X_raw, y_raw = load_data()
    n_samples = len(y_raw)
    train_idx, val_idx, test_idx = get_train_val_test_indices(n_samples)

    # 训练集构建 (使用全部训练数据学习物理规律)
    train_full_idx = np.concatenate([train_idx, val_idx])
    # 输入: [质量, 开度]
    X_train_phys = np.stack([y_raw[train_full_idx], X_raw[train_full_idx, 0]], axis=1)
    # 目标: [转速]
    y_train_phys = X_raw[train_full_idx, 1]

    # 归一化
    x_all = np.stack([y_raw, X_raw[:, 0]], axis=1)

    x_min = x_all[train_full_idx].min(axis=0)
    x_max = x_all[train_full_idx].max(axis=0)
    y_min = X_raw[train_full_idx, 1].min()
    y_max = X_raw[train_full_idx, 1].max()

    X_train_norm = (X_train_phys - x_min) / (x_max - x_min)
    y_train_norm = (y_train_phys - y_min) / (y_max - y_min)

    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).reshape(-1, 1)

    # --- B. 训练 ---
    print(">>> 正在训练反向模型...")
    model = InverseKAN()
    optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # --- C. 筛选验证 (核心修改部分) ---
    print("\n>>> 开始验证 (仅筛选符合策略的样本)...")
    model.eval()

    # 获取测试集原始数据
    test_mass = y_raw[test_idx]
    test_opening = X_raw[test_idx, 0]
    test_speed_true = X_raw[test_idx, 1]

    valid_indices = []  # 存储符合策略的样本索引

    for i in range(len(test_mass)):
        m = test_mass[i]  # 目标质量
        real_op = test_opening[i]  # 实际使用的开度

        # 1. 运行策略：应该用什么开度？
        strat_op = select_optimal_opening(m)

        # 2. 只有当 "实际开度 == 策略开度" 时，才纳入统计
        if np.isclose(real_op, strat_op, atol=0.1):
            valid_indices.append(i)

    # 仅对筛选后的样本进行预测和评估
    if len(valid_indices) > 0:
        f_mass = test_mass[valid_indices]
        f_op = test_opening[valid_indices]  # 这些全都是符合策略的
        f_spd_true = test_speed_true[valid_indices]

        # 预测
        input_vec = np.stack([f_mass, f_op], axis=1)
        input_norm = (input_vec - x_min) / (x_max - x_min)
        input_t = torch.tensor(input_norm, dtype=torch.float32)

        with torch.no_grad():
            pred_norm = model(input_t).numpy().flatten()

        pred_spd = pred_norm * (y_max - y_min) + y_min

        # 计算指标
        r2 = r2_score(f_spd_true, pred_spd)
        mrs = mean_relative_error(f_spd_true, pred_spd)

        print(f"\n===== 策略一致性样本评估 (共 {len(valid_indices)} 个) =====")
        print(f"R² Score: {r2:.4f}")
        print(f"平均相对误差 (MRS): {mrs:.4f}%")

        print("\n[典型样本详情]")
        print(
            f"{'目标质量':<10} | {'策略开度':<10} | {'实际开度':<10} | {'预测转速':<10} | {'实际转速':<10} | {'误差(%)'}")
        for k in range(min(5, len(f_mass))):
            err = abs(pred_spd[k] - f_spd_true[k]) / f_spd_true[k] * 100
            print(
                f"{f_mass[k]:<10.1f} | {f_op[k]:<10.1f} | {f_op[k]:<10.1f} | {pred_spd[k]:<10.2f} | {f_spd_true[k]:<10.2f} | {err:.2f}%")

    else:
        print("测试集中未找到符合当前策略 (20/35/50mm) 的样本。")