# -*- coding: utf-8 -*-
"""
排肥系统交互控制终端（与论文实验口径尽量一致）

功能：
1. 自动在代码同级目录下创建 'path' 文件夹
2. 将训练好的模型权重保存到 'path' 文件夹中
3. 启动时优先从 'path' 文件夹加载模型
4. 模型结构和训练参数优先从 model_meta.json 读取，避免硬编码“论文当前最优参数”
5. 正向模型使用 train_kan.py 中的 FertilizerKAN
6. 反向模型使用 inverse_kan.py 中的 InverseFertilizerKAN
7. 归一化参数基于 train+val 统计，与实验脚本口径一致
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common_utils import load_data, get_train_val_test_indices
from train_kan import FertilizerKAN
from inverse_kan import InverseKANModel, select_optimal_opening


# ================= 配置区域 =================
SAVE_DIR = "path"

MODEL_FWD_PATH = os.path.join(SAVE_DIR, "kan_forward.pth")
MODEL_INV_PATH = os.path.join(SAVE_DIR, "kan_inverse.pth")
META_PATH = os.path.join(SAVE_DIR, "model_meta.json")

DEFAULT_CONFIG = {
    "seed": 42,
    "data_path": "data/dataset.xlsx",
    "normalization_scope": "train+val",

    "forward_model_class": "FertilizerKAN",
    "forward_hidden_dim": 16,
    "forward_lr": 0.005,
    "forward_weight_decay": 1e-5,
    "forward_epochs": 600,

    "inverse_model_class": "InverseFertilizerKAN",
    "inverse_hidden_dim": 16,
    "inverse_lr": 0.005,
    "inverse_weight_decay": 1e-5,
    "inverse_epochs": 600,

    "lr_gamma": 0.99,
}


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


class FertilizerSystem:
    def __init__(self, data_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在初始化施肥控制系统 (Device: {self.device})...")

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.config = self._load_or_init_config(data_path=data_path)
        self.seed = int(self.config["seed"])
        set_seed(self.seed)

        self.data_path = self.config["data_path"]

        # 1. 加载数据并基于 train+val 初始化归一化参数
        self._init_data_params(self.data_path)

        # 2. 获取模型（优先加载，否则训练）
        self.forward_model = self._get_forward_model()
        self.inverse_model = self._get_inverse_model()

        # 3. 保存元信息，确保当前系统实际配置可追踪
        self._save_meta()

        print("\n>>> 系统就绪！当前模型与配置已保存至 path/ 目录")

    # =========================
    # 配置管理
    # =========================
    def _load_or_init_config(self, data_path=None):
        config = dict(DEFAULT_CONFIG)

        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    config.update(loaded)
                print(f"已读取模型配置: {META_PATH}")
            except Exception as e:
                print(f"读取配置失败（{e}），将使用默认配置。")
        else:
            print("未找到 model_meta.json，将使用默认配置初始化。")

        if data_path is not None:
            config["data_path"] = data_path

        return config

    def _save_meta(self):
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def show_config(self):
        print("\n===== 当前系统配置 =====")
        for k, v in self.config.items():
            print(f"{k}: {v}")

    # =========================
    # 数据与归一化参数
    # =========================
    def _init_data_params(self, path):
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
        hidden_dim = int(self.config["forward_hidden_dim"])
        return FertilizerKAN(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=1
        ).to(self.device)

    def _build_inverse_model(self):
        hidden_dim = int(self.config["inverse_hidden_dim"])
        return InverseKANModel(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=1
        ).to(self.device)

    # =========================
    # 模型获取 / 训练 / 保存
    # =========================
    def _get_forward_model(self, force_retrain=False):
        model = self._build_forward_model()

        if os.path.exists(MODEL_FWD_PATH) and not force_retrain:
            print(f"[- 正向模型] 从 {MODEL_FWD_PATH} 加载...")
            try:
                model.load_state_dict(torch.load(MODEL_FWD_PATH, map_location=self.device))
                model.eval()
                return model
            except Exception as e:
                print(f"加载失败（{e}），准备重新训练...")

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
                print(f"加载失败（{e}），准备重新训练...")

        print("[- 反向模型] 正在训练...")
        self._train_model_process(model, mode="inverse")
        torch.save(model.state_dict(), MODEL_INV_PATH)
        print(f"   模型已保存至 -> {MODEL_INV_PATH}")
        return model

    def _train_model_process(self, model, mode="forward"):
        """按与论文实验一致的 train+val 口径训练模型"""
        set_seed(self.seed)

        train_full_idx = np.concatenate([self.idx_tr, self.idx_val])

        if mode == "forward":
            X_train_full = self.raw_X[train_full_idx]
            y_train_full = self.raw_y[train_full_idx]

            X_data = self._norm_forward_x(X_train_full)
            y_data = (y_train_full - self.y_min) / (self.y_max - self.y_min + 1e-8)

            lr = float(self.config["forward_lr"])
            wd = float(self.config["forward_weight_decay"])
            epochs = int(self.config["forward_epochs"])

        elif mode == "inverse":
            inv_X = np.stack([self.raw_y, self.raw_X[:, 0]], axis=1)
            inv_y = self.raw_X[:, 1]

            inv_X_train_full = inv_X[train_full_idx]
            inv_y_train_full = inv_y[train_full_idx]

            X_data = self._norm_inverse_x(inv_X_train_full)
            y_data = self._norm_inverse_y(inv_y_train_full)

            lr = float(self.config["inverse_lr"])
            wd = float(self.config["inverse_weight_decay"])
            epochs = int(self.config["inverse_epochs"])
        else:
            raise ValueError(f"未知模式: {mode}")

        gamma = float(self.config.get("lr_gamma", 0.99))

        X_t = torch.tensor(X_data, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
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

    def retrain_all(self):
        """强制重新训练所有模型"""
        print("\n>>> 开始重新训练所有模型...")
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.forward_model = self._get_forward_model(force_retrain=True)
        self.inverse_model = self._get_inverse_model(force_retrain=True)
        self._save_meta()
        print(">>> 所有模型已更新并保存！")

    # =========================
    # 业务接口
    # =========================
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
        print("\n" + "=" * 48)
        print("   排肥控制系统 v2.1 (Config-Driven)")
        print("=" * 48)
        print("1. [正向预测] 开度/转速 -> 产量")
        print("2. [智能控制] 目标产量 -> 开度/转速")
        print("3. [系统维护] 重新训练模型")
        print("4. [系统维护] 查看当前配置")
        print("5. 退出")

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
            sys_ctrl.show_config()

        elif choice == "5":
            print("系统已退出。")
            break

        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()