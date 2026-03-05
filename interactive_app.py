# -*- coding: utf-8 -*-
"""
排肥系统交互控制终端 (支持模型保存至 path 文件夹)
功能：
1. 自动在代码同级目录下创建 'path' 文件夹
2. 将训练好的模型权重保存到 'path' 文件夹中
3. 启动时优先从 'path' 文件夹加载模型
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import sys
import os  # 用于路径管理

# 导入你现有的模块
from common_utils import load_data
from train_kan import FertilizerKAN
from inverse_kan import InverseKAN, select_optimal_opening

# ================= 配置区域 =================
# 定义保存文件夹名称
SAVE_DIR = "path"
# 定义模型文件路径 (自动拼接文件夹路径)
MODEL_FWD_PATH = os.path.join(SAVE_DIR, "kan_forward.pth")
MODEL_INV_PATH = os.path.join(SAVE_DIR, "kan_inverse.pth")


# ===========================================

class FertilizerSystem:
    def __init__(self, data_path="data/dataset.xlsx"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在初始化施肥控制系统 (Device: {self.device})...")

        # 0. 确保 path 文件夹存在
        if not os.path.exists(SAVE_DIR):
            print(f"创建配置文件夹: {SAVE_DIR}/")
            os.makedirs(SAVE_DIR, exist_ok=True)

        # 1. 加载并分析数据（计算归一化参数）
        self.data_path = data_path
        self._init_data_params(data_path)

        # 2. 获取模型 (加载或训练)
        self.forward_model = self._get_forward_model()
        self.inverse_model = self._get_inverse_model()
        print("\n>>> 系统就绪！所有配置已保存至 path/ 目录")

    def _init_data_params(self, path):
        """计算归一化所需的极值"""
        print(f"正在读取数据: {path} ...")
        # 简单的文件存在性检查
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")

        X, y = load_data(path)

        # 正向模型参数
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self.y_min = y.min()
        self.y_max = y.max()
        self.raw_X = X
        self.raw_y = y

        # 反向模型参数 (输入: [Mass, Opening], 输出: [Speed])
        inv_X_raw = np.stack([y, X[:, 0]], axis=1)
        inv_y_raw = X[:, 1]
        self.inv_x_min = inv_X_raw.min(axis=0)
        self.inv_x_max = inv_X_raw.max(axis=0)
        self.inv_y_min = inv_y_raw.min()
        self.inv_y_max = inv_y_raw.max()

    def _get_forward_model(self, force_retrain=False):
        """获取正向模型"""
        model = FertilizerKAN(input_dim=2, hidden_dim=8, output_dim=1).to(self.device)

        # 检查 path/kan_forward.pth 是否存在
        if os.path.exists(MODEL_FWD_PATH) and not force_retrain:
            print(f"[- 正向模型] 从 {MODEL_FWD_PATH} 加载...")
            try:
                model.load_state_dict(torch.load(MODEL_FWD_PATH, map_location=self.device))
                model.eval()
                return model
            except Exception as e:
                print(f"加载失败 ({e})，准备重新训练...")

        # 否则：训练并保存
        print("[- 正向模型] 正在训练...")
        self._train_model_process(model, mode='forward')
        torch.save(model.state_dict(), MODEL_FWD_PATH)
        print(f"   模型已保存至 -> {MODEL_FWD_PATH}")
        return model

    def _get_inverse_model(self, force_retrain=False):
        """获取反向模型"""
        model = InverseKAN().to(self.device)

        # 检查 path/kan_inverse.pth 是否存在
        if os.path.exists(MODEL_INV_PATH) and not force_retrain:
            print(f"[- 反向模型] 从 {MODEL_INV_PATH} 加载...")
            try:
                model.load_state_dict(torch.load(MODEL_INV_PATH, map_location=self.device))
                model.eval()
                return model
            except Exception as e:
                print(f"加载失败 ({e})，准备重新训练...")

        print("[- 反向模型] 正在训练...")
        self._train_model_process(model, mode='inverse')
        torch.save(model.state_dict(), MODEL_INV_PATH)
        print(f"   模型已保存至 -> {MODEL_INV_PATH}")
        return model

    def _train_model_process(self, model, mode='forward'):
        """通用的训练循环逻辑"""
        # 数据准备
        if mode == 'forward':
            X_data = (self.raw_X - self.X_min) / (self.X_max - self.X_min + 1e-8)
            y_data = (self.raw_y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        else:
            # 反向模型构建数据
            inv_X = np.stack([self.raw_y, self.raw_X[:, 0]], axis=1)
            inv_y = self.raw_X[:, 1]
            X_data = (inv_X - self.inv_x_min) / (self.inv_x_max - self.inv_x_min + 1e-8)
            y_data = (inv_y - self.inv_y_min) / (self.inv_y_max - self.inv_y_min + 1e-8)

        X_t = torch.tensor(X_data, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.MSELoss()

        model.train()
        # 训练 300 轮 (可根据需要调整)
        for epoch in range(300):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

        model.eval()

    def retrain_all(self):
        """强制重新训练所有模型"""
        print("\n>>> 开始重新训练所有模型...")
        # 确保目录存在
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.forward_model = self._get_forward_model(force_retrain=True)
        self.inverse_model = self._get_inverse_model(force_retrain=True)
        print(">>> 所有模型已更新并保存！")

    def predict_mass(self, opening, speed):
        """正向预测"""
        input_arr = np.array([[opening, speed]], dtype=np.float32)
        input_norm = (input_arr - self.X_min) / (self.X_max - self.X_min + 1e-8)
        input_t = torch.tensor(input_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_norm = self.forward_model(input_t).cpu().numpy().item()
        return pred_norm * (self.y_max - self.y_min) + self.y_min

    def intelligent_control(self, target_mass):
        """智能控制"""
        rec_opening = select_optimal_opening(target_mass)
        input_arr = np.array([[target_mass, rec_opening]], dtype=np.float32)
        input_norm = (input_arr - self.inv_x_min) / (self.inv_x_max - self.inv_x_min + 1e-8)
        input_t = torch.tensor(input_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_speed_norm = self.inverse_model(input_t).cpu().numpy().item()
        pred_speed = pred_speed_norm * (self.inv_y_max - self.inv_y_min) + self.inv_y_min
        return rec_opening, pred_speed


def main():
    try:
        sys_ctrl = FertilizerSystem()
    except Exception as e:
        print(f"初始化错误: {e}")
        return

    while True:
        print("\n" + "=" * 40)
        print("   排肥控制系统 v1.2 (Path Integrated)")
        print("=" * 40)
        print("1. [正向预测] 开度/转速 -> 产量")
        print("2. [智能控制] 目标产量 -> 开度/转速")
        print("3. [系统维护] 重新训练模型")
        print("4. 退出")

        choice = input("请输入选项: ").strip()

        if choice == '1':
            try:
                op = float(input("请输入开度 (mm): "))
                spd = float(input("请输入转速 (r/min): "))
                res = sys_ctrl.predict_mass(op, spd)
                print(f"\n✅ 预测排肥量: {res:.2f} g/min")
            except ValueError:
                print("输入错误")

        elif choice == '2':
            try:
                target = float(input("请输入目标产量 (g/min): "))
                op, spd = sys_ctrl.intelligent_control(target)
                print(f"\n✅ 推荐策略:")
                print(f"   - 调节开度: {op} mm")
                print(f"   - 设定转速: {spd:.2f} r/min")
            except ValueError:
                print("输入错误")

        elif choice == '3':
            confirm = input("确定要重新训练吗？(y/n): ")
            if confirm.lower() == 'y':
                sys_ctrl.retrain_all()

        elif choice == '4':
            break


if __name__ == "__main__":
    main()