# -*- coding: utf-8 -*-
"""
排肥系统交互控制终端（优先消费训练工件，精简配置版）

功能：
1. 自动在代码同级目录下创建 'path' 文件夹
2. 启动时优先从 'path' 文件夹加载模型
3. 优先读取训练脚本落盘的 forward / inverse 工件元数据
4. 若工件缺失，则回退到“从原始数据重算归一化参数 + fallback 训练参数”的旧逻辑
5. 正向模型使用 train_kan.py 中的 FertilizerKAN
6. 反向模型使用 inverse_kan.py 中的 InverseKANModel
7. 正向模型归一化参数基于 train；反向模型归一化参数基于 train+val
8. 增加输入越界检查、外推告警与输出裁剪，提升交互系统稳健性
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
FORWARD_META_PATH = os.path.join(SAVE_DIR, "kan_forward_meta.json")
INVERSE_META_PATH = os.path.join(SAVE_DIR, "kan_inverse_meta.json")

# 只保留“交互系统配置”
DEFAULT_CONFIG = {
    "seed": 42,
    "data_path": "data/dataset.xlsx",
    "enable_range_warning": True,
    "enable_output_clipping": True,
    "prefer_saved_artifacts": True,
}

# 仅用于“工件缺失时的回退训练”
FALLBACK_MODEL_PARAMS = {
    "forward_hidden_dim": 16,
    "forward_lr": 0.005,
    "forward_weight_decay": 1e-5,
    "forward_epochs": 600,

    "inverse_hidden_dim": 16,
    "inverse_lr": 0.005,
    "inverse_weight_decay": 1e-5,
    "inverse_epochs": 1000,

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

        # 只加载交互配置
        self.config = self._load_or_init_config(data_path=data_path)
        self.seed = int(self.config["seed"])
        set_seed(self.seed)

        self.data_path = self.config["data_path"]

        # 运行期模型参数：优先由工件恢复；否则使用 fallback
        self.model_params = dict(FALLBACK_MODEL_PARAMS)

        # 标记是否成功从工件恢复
        self.loaded_forward_artifact = False
        self.loaded_inverse_artifact = False

        # 1. 优先从工件恢复归一化参数与边界；若失败则回退到原始数据重算
        self._init_runtime_state()

        # 2. 获取模型（优先加载，否则训练）
        self.forward_model = self._get_forward_model()
        self.inverse_model = self._get_inverse_model()

        # 3. 保存精简后的系统配置
        self._save_meta()

        print("\n>>> 系统就绪！当前系统配置已保存至 path/model_meta.json")

    # =========================
    # 配置管理（仅交互配置）
    # =========================
    def _load_or_init_config(self, data_path=None):
        config = dict(DEFAULT_CONFIG)

        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    # 只接受 DEFAULT_CONFIG 中定义的键，自动忽略历史冗余字段
                    for k in DEFAULT_CONFIG.keys():
                        if k in loaded:
                            config[k] = loaded[k]
                print(f"已读取模型配置: {META_PATH}")
            except Exception as e:
                print(f"读取配置失败（{e}），将使用默认配置。")
        else:
            print("未找到 model_meta.json，将使用默认配置初始化。")

        if data_path is not None:
            config["data_path"] = data_path

        return config

    def _save_meta(self):
        # 只保存精简后的交互配置
        clean_config = {k: self.config[k] for k in DEFAULT_CONFIG.keys()}
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(clean_config, f, ensure_ascii=False, indent=2)

    def show_config(self):
        print("\n===== 当前交互配置 =====")
        for k, v in self.config.items():
            print(f"{k}: {v}")

        print("\n===== 当前运行状态 =====")
        print(f"forward_artifact_loaded: {self.loaded_forward_artifact}")
        print(f"inverse_artifact_loaded: {self.loaded_inverse_artifact}")
        print(f"forward_weight_exists: {os.path.exists(MODEL_FWD_PATH)}")
        print(f"inverse_weight_exists: {os.path.exists(MODEL_INV_PATH)}")
        print(f"forward_meta_exists: {os.path.exists(FORWARD_META_PATH)}")
        print(f"inverse_meta_exists: {os.path.exists(INVERSE_META_PATH)}")

        print("\n===== 当前模型参数（运行期） =====")
        for k, v in self.model_params.items():
            print(f"{k}: {v}")

    # =========================
    # 初始化运行态：优先工件，回退原始数据
    # =========================
    def _init_runtime_state(self):
        prefer_artifacts = bool(self.config.get("prefer_saved_artifacts", True))

        if prefer_artifacts:
            fwd_ok = self._try_load_forward_artifact()
            inv_ok = self._try_load_inverse_artifact()

            if fwd_ok and inv_ok:
                print("已从 forward / inverse 工件恢复归一化参数与训练域边界。")
                return

            print("工件恢复不完整，回退到原始数据重算归一化参数。")

        self._init_data_params_from_raw_data(self.data_path)

    def _try_load_forward_artifact(self):
        if not os.path.exists(FORWARD_META_PATH):
            return False

        try:
            with open(FORWARD_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

            norm_params = meta["normalization_params"]
            domain = meta["training_domain"]
            hyper = meta.get("hyperparameters", {})

            self.X_min = np.asarray(norm_params["X_min"], dtype=np.float32)
            self.X_max = np.asarray(norm_params["X_max"], dtype=np.float32)
            self.y_min = float(norm_params["y_min"])
            self.y_max = float(norm_params["y_max"])

            self.forward_opening_min = float(domain["opening_min"])
            self.forward_opening_max = float(domain["opening_max"])
            self.forward_speed_min = float(domain["speed_min"])
            self.forward_speed_max = float(domain["speed_max"])
            self.forward_mass_min = float(domain["mass_min"])
            self.forward_mass_max = float(domain["mass_max"])

            if "hidden_dim" in hyper:
                self.model_params["forward_hidden_dim"] = int(hyper["hidden_dim"])
            if "lr" in hyper:
                self.model_params["forward_lr"] = float(hyper["lr"])
            if "weight_decay" in hyper:
                self.model_params["forward_weight_decay"] = float(hyper["weight_decay"])
            if "epochs" in hyper:
                self.model_params["forward_epochs"] = int(hyper["epochs"])

            self.loaded_forward_artifact = True
            print(f"已读取正向工件: {FORWARD_META_PATH}")
            return True

        except Exception as e:
            print(f"读取正向工件失败（{e}），将回退到原始数据。")
            self.loaded_forward_artifact = False
            return False

    def _try_load_inverse_artifact(self):
        if not os.path.exists(INVERSE_META_PATH):
            return False

        try:
            with open(INVERSE_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

            norm_params = meta["normalization_params"]
            domain = meta["training_domain"]
            hyper = meta.get("hyperparameters", {})

            self.inv_x_min = np.asarray(norm_params["X_min"], dtype=np.float32)
            self.inv_x_max = np.asarray(norm_params["X_max"], dtype=np.float32)
            self.inv_y_min = float(norm_params["y_min"])
            self.inv_y_max = float(norm_params["y_max"])

            self.inverse_target_mass_min = float(domain["target_mass_min"])
            self.inverse_target_mass_max = float(domain["target_mass_max"])
            self.inverse_opening_min = float(domain["opening_min"])
            self.inverse_opening_max = float(domain["opening_max"])
            self.inverse_speed_min = float(domain["speed_min"])
            self.inverse_speed_max = float(domain["speed_max"])

            if "hidden_dim" in hyper:
                self.model_params["inverse_hidden_dim"] = int(hyper["hidden_dim"])
            if "lr" in hyper:
                self.model_params["inverse_lr"] = float(hyper["lr"])
            if "weight_decay" in hyper:
                self.model_params["inverse_weight_decay"] = float(hyper["weight_decay"])
            if "epochs" in hyper:
                self.model_params["inverse_epochs"] = int(hyper["epochs"])

            self.loaded_inverse_artifact = True
            print(f"已读取反向工件: {INVERSE_META_PATH}")
            return True

        except Exception as e:
            print(f"读取反向工件失败（{e}），将回退到原始数据。")
            self.loaded_inverse_artifact = False
            return False

    # =========================
    # 回退逻辑：从原始数据重算
    # =========================
    def _init_data_params_from_raw_data(self, path):
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

        forward_norm_idx = idx_tr
        inverse_norm_idx = np.concatenate([idx_tr, idx_val])

        # ---------- 正向模型归一化参数（仅基于 train） ----------
        X_train = X[forward_norm_idx]
        y_train = y[forward_norm_idx]

        self.X_min = X_train.min(axis=0, keepdims=True)
        self.X_max = X_train.max(axis=0, keepdims=True)
        self.y_min = float(y_train.min())
        self.y_max = float(y_train.max())

        self.forward_opening_min = float(X_train[:, 0].min())
        self.forward_opening_max = float(X_train[:, 0].max())
        self.forward_speed_min = float(X_train[:, 1].min())
        self.forward_speed_max = float(X_train[:, 1].max())
        self.forward_mass_min = float(y_train.min())
        self.forward_mass_max = float(y_train.max())

        # ---------- 反向模型归一化参数（基于 train+val） ----------
        inv_X_raw = np.stack([y, X[:, 0]], axis=1)
        inv_y_raw = X[:, 1]

        inv_X_train_full = inv_X_raw[inverse_norm_idx]
        inv_y_train_full = inv_y_raw[inverse_norm_idx]

        self.inv_x_min = inv_X_train_full.min(axis=0, keepdims=True)
        self.inv_x_max = inv_X_train_full.max(axis=0, keepdims=True)
        self.inv_y_min = float(inv_y_train_full.min())
        self.inv_y_max = float(inv_y_train_full.max())

        self.inverse_target_mass_min = float(inv_X_train_full[:, 0].min())
        self.inverse_target_mass_max = float(inv_X_train_full[:, 0].max())
        self.inverse_opening_min = float(inv_X_train_full[:, 1].min())
        self.inverse_opening_max = float(inv_X_train_full[:, 1].max())
        self.inverse_speed_min = float(inv_y_train_full.min())
        self.inverse_speed_max = float(inv_y_train_full.max())

        print("已基于原始数据重算归一化参数与训练域边界。")

    # =========================
    # 归一化 / 反归一化工具
    # =========================
    def _norm_forward_x(self, x):
        return (x - self.X_min) / (self.X_max - self.X_min + 1e-8)

    def _denorm_forward_y(self, y_norm):
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min

    def _norm_inverse_x(self, x):
        return (x - self.inv_x_min) / (self.inv_x_max - self.inv_x_min + 1e-8)

    def _denorm_inverse_y(self, y_norm):
        return y_norm * (self.inv_y_max - self.inv_y_min + 1e-8) + self.inv_y_min

    # =========================
    # 交互保护工具
    # =========================
    def _format_range(self, low, high, unit=""):
        return f"[{low:.2f}, {high:.2f}]{unit}"

    def _check_forward_input_range(self, opening, speed):
        warnings = []

        if opening < self.forward_opening_min or opening > self.forward_opening_max:
            warnings.append(
                f"开度 {opening:.2f} mm 超出正向模型训练范围 "
                f"{self._format_range(self.forward_opening_min, self.forward_opening_max, ' mm')}"
            )

        if speed < self.forward_speed_min or speed > self.forward_speed_max:
            warnings.append(
                f"转速 {speed:.2f} r/min 超出正向模型训练范围 "
                f"{self._format_range(self.forward_speed_min, self.forward_speed_max, ' r/min')}"
            )

        return warnings

    def _check_inverse_input_range(self, target_mass, opening):
        warnings = []

        if target_mass < self.inverse_target_mass_min or target_mass > self.inverse_target_mass_max:
            warnings.append(
                f"目标排肥量 {target_mass:.2f} g/min 超出反向模型训练范围 "
                f"{self._format_range(self.inverse_target_mass_min, self.inverse_target_mass_max, ' g/min')}"
            )

        if opening < self.inverse_opening_min or opening > self.inverse_opening_max:
            warnings.append(
                f"推荐开度 {opening:.2f} mm 超出反向模型训练范围 "
                f"{self._format_range(self.inverse_opening_min, self.inverse_opening_max, ' mm')}"
            )

        return warnings

    def _clip_speed_to_physical_range(self, speed):
        clipped_speed = float(np.clip(speed, self.inverse_speed_min, self.inverse_speed_max))
        was_clipped = not np.isclose(clipped_speed, float(speed))
        return clipped_speed, was_clipped

    def _clip_mass_to_physical_range(self, mass):
        clipped_mass = float(np.clip(mass, self.forward_mass_min, self.forward_mass_max))
        was_clipped = not np.isclose(clipped_mass, float(mass))
        return clipped_mass, was_clipped

    # =========================
    # 模型定义
    # =========================
    def _build_forward_model(self):
        hidden_dim = int(self.model_params["forward_hidden_dim"])
        return FertilizerKAN(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=1
        ).to(self.device)

    def _build_inverse_model(self):
        hidden_dim = int(self.model_params["inverse_hidden_dim"])
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
        """
        回退训练逻辑：
        当缺少可用工件时，使用原始数据按论文实验口径训练。
        """
        if not hasattr(self, "raw_X") or not hasattr(self, "raw_y"):
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    f"缺少可用工件且找不到原始数据文件: {self.data_path}"
                )
            self._init_data_params_from_raw_data(self.data_path)

        set_seed(self.seed)
        train_full_idx = np.concatenate([self.idx_tr, self.idx_val])

        if mode == "forward":
            X_train_full = self.raw_X[train_full_idx]
            y_train_full = self.raw_y[train_full_idx]

            X_data = self._norm_forward_x(X_train_full)
            y_data = (y_train_full - self.y_min) / (self.y_max - self.y_min + 1e-8)

            lr = float(self.model_params["forward_lr"])
            wd = float(self.model_params["forward_weight_decay"])
            epochs = int(self.model_params["forward_epochs"])

        elif mode == "inverse":
            inv_X = np.stack([self.raw_y, self.raw_X[:, 0]], axis=1)
            inv_y = self.raw_X[:, 1]

            inv_X_train_full = inv_X[train_full_idx]
            inv_y_train_full = inv_y[train_full_idx]

            X_data = self._norm_inverse_x(inv_X_train_full)
            y_data = (inv_y_train_full - self.inv_y_min) / (self.inv_y_max - self.inv_y_min + 1e-8)

            lr = float(self.model_params["inverse_lr"])
            wd = float(self.model_params["inverse_weight_decay"])
            epochs = int(self.model_params["inverse_epochs"])
        else:
            raise ValueError(f"未知模式: {mode}")

        gamma = float(self.model_params.get("lr_gamma", 0.99))

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
        print("\n>>> 开始重新训练所有模型...")
        os.makedirs(SAVE_DIR, exist_ok=True)

        self._init_data_params_from_raw_data(self.data_path)

        self.forward_model = self._get_forward_model(force_retrain=True)
        self.inverse_model = self._get_inverse_model(force_retrain=True)
        self._save_meta()
        print(">>> 所有模型已更新并保存！")

    # =========================
    # 业务接口
    # =========================
    def predict_mass(self, opening, speed):
        warnings = []
        if self.config.get("enable_range_warning", True):
            warnings.extend(self._check_forward_input_range(opening, speed))

        input_arr = np.array([[opening, speed]], dtype=np.float32)
        input_norm = self._norm_forward_x(input_arr)
        input_t = torch.tensor(input_norm, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_norm = self.forward_model(input_t).cpu().numpy().item()

        pred_mass = float(self._denorm_forward_y(pred_norm))

        clipped = False
        if self.config.get("enable_output_clipping", True):
            pred_mass, clipped = self._clip_mass_to_physical_range(pred_mass)

        return {
            "pred_mass": float(pred_mass),
            "warnings": warnings,
            "clipped": clipped
        }

    def intelligent_control(self, target_mass):
        rec_opening = float(select_optimal_opening(target_mass))

        warnings = []
        if self.config.get("enable_range_warning", True):
            warnings.extend(self._check_inverse_input_range(target_mass, rec_opening))

        input_arr = np.array([[target_mass, rec_opening]], dtype=np.float32)
        input_norm = self._norm_inverse_x(input_arr)
        input_t = torch.tensor(input_norm, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_speed_norm = self.inverse_model(input_t).cpu().numpy().item()

        pred_speed = float(self._denorm_inverse_y(pred_speed_norm))

        clipped = False
        if self.config.get("enable_output_clipping", True):
            pred_speed, clipped = self._clip_speed_to_physical_range(pred_speed)

        return {
            "opening": rec_opening,
            "speed": float(pred_speed),
            "warnings": warnings,
            "clipped": clipped
        }


def main():
    try:
        sys_ctrl = FertilizerSystem()
    except Exception as e:
        print(f"初始化错误: {e}")
        return

    while True:
        print("\n" + "=" * 52)
        print("   排肥控制系统 v2.4 (Lean Config)")
        print("=" * 52)
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

                result = sys_ctrl.predict_mass(op, spd)

                print(f"\n✅ 预测排肥量: {result['pred_mass']:.2f} g/min")

                if result["warnings"]:
                    print("⚠ 输入超出训练范围，当前结果属于外推预测，可信度可能下降：")
                    for msg in result["warnings"]:
                        print(f"   - {msg}")

                if result["clipped"]:
                    print("⚠ 预测结果已裁剪到训练数据支持的物理范围内。")

            except ValueError:
                print("输入错误，请输入数值。")

        elif choice == "2":
            try:
                target = float(input("请输入目标产量 (g/min): "))

                result = sys_ctrl.intelligent_control(target)

                print("\n✅ 推荐策略:")
                print(f"   - 调节开度: {result['opening']:.1f} mm")
                print(f"   - 设定转速: {result['speed']:.2f} r/min")

                if result["warnings"]:
                    print("⚠ 输入超出训练范围，当前结果属于外推控制建议，可信度可能下降：")
                    for msg in result["warnings"]:
                        print(f"   - {msg}")

                if result["clipped"]:
                    print("⚠ 推荐转速已裁剪到训练数据支持的物理范围内。")

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