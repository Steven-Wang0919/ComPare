# -*- coding: utf-8 -*-
"""
排肥系统交互控制终端（与论文实验口径尽量一致）

功能：
1. 自动在代码同级目录下创建 'path' 文件夹
2. 将训练好的模型权重保存到 'path' 文件夹中
3. 启动时优先从 'path' 文件夹加载模型
4. 正向模型使用论文当前最优 KAN 结构默认值
5. 反向模型使用 inverse_kan_V2 中的最终版本模型
6. 归一化参数基于 train+val 统计，与实验脚本口径一致
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common_utils import load_data, get_train_val_test_indices
from train_kan import FertilizerKAN
from inverse_kan_V2 import InverseFertilizerKAN, select_optimal_opening


# ================= 配置区域 =================
SAVE_DIR = "path"

MODEL_FWD_PATH = os.path.join(SAVE_DIR, "kan_forward.pth")
MODEL_INV_PATH = os.path.join(SAVE_DIR, "kan_inverse.pth")
META_PATH = os.path.join(SAVE_DIR, "model_meta.json")

DEFAULT_SEED = 42

# 与论文当前结果保持一致的默认结构
FORWARD_HIDDEN_DIM = 16
INVERSE_HIDDEN_DIM = 16

# 训练参数
FORWARD_LR = 0.005
FORWARD_WEIGHT_DECAY = 1e-5

INVERSE_LR = 0.005
INVERSE_WEIGHT_DECAY = 1e-5

FORWARD_EPOCHS = 600
INVERSE_EPOCHS = 600
LR_GAMMA = 0.99


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FertilizerSystem:
    def __init__(self, data_path="data/dataset.xlsx"):
        set_seed(DEFAULT_SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在初始化施肥控制系统 (Device: {self.device})...")

        if not os.path.exists(SAVE_DIR):
            print(f"创建配置文件夹: {SAVE_DIR}/")
            os.makedirs(SAVE_DIR, exist_ok=True)

        self.data_path = data_path

        # 1. 加载数据并基于 train+val 初始化归一化参数
        self._init_data_params(data_path)

        # 2. 获取模型（优先加载，否则训练）
        self.forward_model = self._get_forward_model()
        self.inverse_model = self._get_inverse_model()

        # 3. 保存元信息，确保后续知道当前系统实际加载的是哪套模型
        self._save_meta()

        print("\n>>> 系统就绪！当前模型与配置已保存至 path/ 目录")

    def _init_data_params(self, path):
        """基于 train+val 统计归一化参数，与实验脚本保持一致"""
        print(f"正在读取数据: {path} ...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")

        X, y = load_data(path)
        self.raw_X = X
        self.raw_y = y

        n_samples = len(y)
        idx_tr, idx_val, idx_te = get_train_val_test_indices(n_samples)
        self.idx_tr = idx_tr
        self.idx_val = idx_val
        self.idx_te = idx_te

        train_full_idx = np.concatenate([idx_tr, idx_val])

        # ---------- 正向模型归一化参数 ----------
        X_train_full = X[train_full_idx]
        y_train_full = y[train_full_idx]

        self.X_min = X_train_full.min(axis=0, keepdims=True)
        self.X_max = X_train_full.max(axis=0, keepdims=True)
        self.y_min = float(y_train_full.min())
        self.y_max = float(y_train_full.max())

        # ---------- 反向模型归一化参数 ----------
        inv_X_raw = np.stack([y, X[:, 0]], axis=1)   # [mass, opening]
        inv_y_raw = X[:, 1]                          # [speed]

        inv_X_train_full = inv_X_raw[train_full_idx]
        inv_y_train_full = inv_y_raw[train_full_idx]

        self.inv_x_min = inv_X_train_full.min(axis=0, keepdims=True)
        self.inv_x_max = inv_X_train_full.max(axis=0, keepdims=True)
        self.inv_y_min = float(inv_y_train_full.min())
        self.inv_y_max = float(inv_y_train_full.max())

    # =========================
    # 归一化 / 反归一化工具
    # =========================
    def _norm_forward_x(self, x):
        return (x - self.X_min) / (self.X_max - self.X_min + 1e-8)

    def _denorm_forward_y(self, y_norm):
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min

    def _norm_inverse_x(self, x):
        return (x - self.inv_x_min) / (self.inv_x_max - self.inv_x_min + 1e-8)

    def _norm_inverse_y(self, y):
        return (y - self.inv_y_min) / (self.inv_y_max - self.inv_y_min + 1e-8)

    def _denorm_inverse_y(self, y_norm):
        return y_norm * (self.inv_y_max - self.inv_y_min + 1e-8) + self.inv_y_min

    # =========================
    # 模型定义
    # =========================
    def _build_forward_model(self):
        return FertilizerKAN(
            input_dim=2,
            hidden_dim=FORWARD_HIDDEN_DIM,
            output_dim=1
        ).to(self.device)

    def _build_inverse_model(self):
        return InverseFertilizerKAN(
            input_dim=2,
            hidden_dim=INVERSE_HIDDEN_DIM,
            output_dim=1
        ).to(self.device)

    def _get_forward_model(self, force_retrain=False):
        model = self._build_forward_model()

        if os.path.exists(MODEL_FWD_PATH) and not force_retrain:
            print(f"[- 正向模型] 从 {MODEL_FWD_PATH} 加载...")
            try:
                model.load_state_dict(torch.load(MODEL_FWD_PATH, map_location=self.device))
                model.eval()
                return model
            except Exception as e:
                print(f"加载失败 ({e})，准备重新训练...")

        print("[- 正向模型] 正在训练...")
        self._train_model_process(model, mode="forward")
        torch.save(model.state_dict(), MODEL_FWD_PATH)
        print(f"   模型已保存至 -> {MODEL_FWD_PATH}")
        return model

    def _get_inverse_model(self, force_retrain=False):
        model = self._build_inverse_model()

        if os.path.exists(MODEL_INV_PATH) and not force_retrain:
            print(f"[- 反向模型] 从 {MODEL_INV_PATH} 加载...")
            try:
                model.load_state_dict(torch.load(MODEL_INV_PATH, map_location=self.device))
                model.eval()
                return model
            except Exception as e:
                print(f"加载失败 ({e})，准备重新训练...")

        print("[- 反向模型] 正在训练...")
        self._train_model_process(model, mode="inverse")
        torch.save(model.state_dict(), MODEL_INV_PATH)
        print(f"   模型已保存至 -> {MODEL_INV_PATH}")
        return model

    def _train_model_process(self, model, mode="forward"):
        """按与论文实验一致的 train+val 口径训练模型"""
        set_seed(DEFAULT_SEED)

        train_full_idx = np.concatenate([self.idx_tr, self.idx_val])

        if mode == "forward":
            X_train_full = self.raw_X[train_full_idx]
            y_train_full = self.raw_y[train_full_idx]

            X_data = self._norm_forward_x(X_train_full)
            y_data = (y_train_full - self.y_min) / (self.y_max - self.y_min + 1e-8)

            lr = FORWARD_LR
            wd = FORWARD_WEIGHT_DECAY
            epochs = FORWARD_EPOCHS

        elif mode == "inverse":
            inv_X = np.stack([self.raw_y, self.raw_X[:, 0]], axis=1)
            inv_y = self.raw_X[:, 1]

            inv_X_train_full = inv_X[train_full_idx]
            inv_y_train_full = inv_y[train_full_idx]

            X_data = self._norm_inverse_x(inv_X_train_full)
            y_data = self._norm_inverse_y(inv_y_train_full)

            lr = INVERSE_LR
            wd = INVERSE_WEIGHT_DECAY
            epochs = INVERSE_EPOCHS
        else:
            raise ValueError(f"未知模式: {mode}")

        X_t = torch.tensor(X_data, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 100 == 0:
                print(f"   [{mode}] Epoch {epoch + 1}/{epochs}, Loss={loss.item():.6f}")

        model.eval()

    def _save_meta(self):
        meta = {
            "forward_model_class": "FertilizerKAN",
            "forward_hidden_dim": FORWARD_HIDDEN_DIM,
            "forward_lr": FORWARD_LR,
            "forward_weight_decay": FORWARD_WEIGHT_DECAY,
            "forward_epochs": FORWARD_EPOCHS,
            "inverse_model_class": "InverseFertilizerKAN",
            "inverse_hidden_dim": INVERSE_HIDDEN_DIM,
            "inverse_lr": INVERSE_LR,
            "inverse_weight_decay": INVERSE_WEIGHT_DECAY,
            "inverse_epochs": INVERSE_EPOCHS,
            "seed": DEFAULT_SEED,
            "data_path": self.data_path,
            "normalization_scope": "train+val",
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def retrain_all(self):
        """强制重新训练所有模型"""
        print("\n>>> 开始重新训练所有模型...")
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.forward_model = self._get_forward_model(force_retrain=True)
        self.inverse_model = self._get_inverse_model(force_retrain=True)
        self._save_meta()
        print(">>> 所有模型已更新并保存！")

    def predict_mass(self, opening, speed):
        """正向预测：开度/转速 -> 排肥量"""
        input_arr = np.array([[opening, speed]], dtype=np.float32)
        input_norm = self._norm_forward_x(input_arr)
        input_t = torch.tensor(input_norm, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_norm = self.forward_model(input_t).cpu().numpy().item()

        return float(self._denorm_forward_y(pred_norm))

    def intelligent_control(self, target_mass):
        """智能控制：目标产量 -> 推荐开度/转速"""
        rec_opening = select_optimal_opening(target_mass)

        input_arr = np.array([[target_mass, rec_opening]], dtype=np.float32)
        input_norm = self._norm_inverse_x(input_arr)
        input_t = torch.tensor(input_norm, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_speed_norm = self.inverse_model(input_t).cpu().numpy().item()

        pred_speed = self._denorm_inverse_y(pred_speed_norm)
        return float(rec_opening), float(pred_speed)


def main():
    try:
        sys_ctrl = FertilizerSystem()
    except Exception as e:
        print(f"初始化错误: {e}")
        return

    while True:
        print("\n" + "=" * 44)
        print("   排肥控制系统 v2.0 (Paper-Aligned)")
        print("=" * 44)
        print("1. [正向预测] 开度/转速 -> 产量")
        print("2. [智能控制] 目标产量 -> 开度/转速")
        print("3. [系统维护] 重新训练模型")
        print("4. 退出")

        choice = input("请输入选项: ").strip()

        if choice == "1":
            try:
                op = float(input("请输入开度 (mm): "))
                spd = float(input("请输入转速 (r/min): "))
                res = sys_ctrl.predict_mass(op, spd)
                print(f"\n✅ 预测排肥量: {res:.2f} g/min")
            except ValueError:
                print("输入错误，请输入数值。")

        elif choice == "2":
            try:
                target = float(input("请输入目标产量 (g/min): "))
                op, spd = sys_ctrl.intelligent_control(target)
                print("\n✅ 推荐策略:")
                print(f"   - 调节开度: {op:.1f} mm")
                print(f"   - 设定转速: {spd:.2f} r/min")
            except ValueError:
                print("输入错误，请输入数值。")

        elif choice == "3":
            confirm = input("确定要重新训练吗？(y/n): ").strip().lower()
            if confirm == "y":
                sys_ctrl.retrain_all()

        elif choice == "4":
            print("系统已退出。")
            break

        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()