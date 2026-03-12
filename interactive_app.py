# -*- coding: utf-8 -*-
"""
排肥系统交互控制终端（runs 工件目录兼容版）

本版重点修复：
1. 不再硬编码只从 path/ 读取工件
2. 启动时自动从以下位置解析工件：
   - path/model_meta.json 中显式配置的工件目录
   - 环境变量 FORWARD_ARTIFACT_DIR / INVERSE_ARTIFACT_DIR
   - runs/ 下最新的 artifacts/
   - 旧版 path/（兼容）
3. 显式重训练时，将 forward / inverse 工件分别保存到 runs/<timestamp>_.../artifacts/
4. 重训练完成后自动回写 path/model_meta.json，后续启动可直接定位到最新工件
5. 保留严格工件模式、输入越界告警、输出裁剪、原始值与裁剪后值并列展示
"""

import json
import os
import random
import tempfile
from datetime import datetime

import numpy as np
import torch

from train_kan import FertilizerKAN, train_and_eval_kan
from inverse_kan import InverseKANModel, select_optimal_opening, train_and_eval_inverse_kan_v2


# -----------------------------------------------------------------------------
# 兼容旧配置：path/ 仍作为配置目录，不再作为唯一工件目录
# -----------------------------------------------------------------------------
CONFIG_DIR = "path"
META_PATH = os.path.join(CONFIG_DIR, "model_meta.json")

LEGACY_FORWARD_META_PATH = os.path.join(CONFIG_DIR, "kan_forward_meta.json")
LEGACY_INVERSE_META_PATH = os.path.join(CONFIG_DIR, "kan_inverse_meta.json")
LEGACY_FORWARD_WEIGHT_PATH = os.path.join(CONFIG_DIR, "kan_forward.pth")
LEGACY_INVERSE_WEIGHT_PATH = os.path.join(CONFIG_DIR, "kan_inverse.pth")

DEFAULT_CONFIG = {
    "seed": 42,
    "data_path": "data/dataset.xlsx",
    "enable_range_warning": True,
    "enable_output_clipping": True,
    "prefer_saved_artifacts": True,
    "strict_artifact_mode": True,

    # 新增：显式记录当前使用的工件目录
    "forward_artifact_dir": None,
    "inverse_artifact_dir": None,
    "runs_root": "runs",
}

# 仅供显式重训练时使用；正式训练脚本会基于候选集自行搜索最优参数
DEFAULT_TRAINING_CONFIG = {
    "forward_hidden_dim_candidates": [4, 8, 16],
    "forward_lr_candidates": [0.01, 0.005],
    "forward_weight_decay_candidates": [1e-4, 1e-5],
    "forward_epochs": 600,
    "forward_search_epochs": 300,
    "forward_lr_gamma": 0.99,

    "inverse_hidden_dim_candidates": [8, 16, 32],
    "inverse_lr_candidates": [1e-2, 5e-3, 1e-3],
    "inverse_weight_decay_candidates": [0.0, 1e-6, 1e-5, 1e-4],
    "inverse_epochs": 1000,
}


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def now_timestamp():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def normpath(path):
    return os.path.normpath(path) if path is not None else None


def safe_exists(path):
    return path is not None and os.path.exists(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_artifact_dir(runs_root, suffix_name):
    """
    在 runs/ 下寻找最新的 <timestamp>_<suffix_name>/artifacts 目录。
    suffix_name 例如：
      - train_kan
      - inverse_kan
    """
    runs_root = normpath(runs_root)
    if not os.path.isdir(runs_root):
        return None

    candidates = []
    for name in os.listdir(runs_root):
        full = os.path.join(runs_root, name)
        if not os.path.isdir(full):
            continue
        if not name.endswith(f"_{suffix_name}"):
            continue

        art_dir = os.path.join(full, "artifacts")
        if os.path.isdir(art_dir):
            candidates.append((name, art_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return normpath(candidates[-1][1])


def resolve_forward_artifact_dir(config):
    """
    正向工件目录优先级：
    1. config["forward_artifact_dir"]
    2. 环境变量 FORWARD_ARTIFACT_DIR
    3. runs/ 下最新 train_kan artifacts
    4. 旧版 path/
    """
    explicit_dir = config.get("forward_artifact_dir")
    if explicit_dir and os.path.isdir(explicit_dir):
        return normpath(explicit_dir)

    env_dir = os.environ.get("FORWARD_ARTIFACT_DIR")
    if env_dir and os.path.isdir(env_dir):
        return normpath(env_dir)

    runs_root = config.get("runs_root", "runs")
    latest = find_latest_artifact_dir(runs_root, "train_kan")
    if latest and os.path.isdir(latest):
        return normpath(latest)

    if os.path.isdir(CONFIG_DIR):
        return normpath(CONFIG_DIR)

    return None


def resolve_inverse_artifact_dir(config):
    """
    反向工件目录优先级：
    1. config["inverse_artifact_dir"]
    2. 环境变量 INVERSE_ARTIFACT_DIR
    3. runs/ 下最新 inverse_kan artifacts
    4. 旧版 path/
    """
    explicit_dir = config.get("inverse_artifact_dir")
    if explicit_dir and os.path.isdir(explicit_dir):
        return normpath(explicit_dir)

    env_dir = os.environ.get("INVERSE_ARTIFACT_DIR")
    if env_dir and os.path.isdir(env_dir):
        return normpath(env_dir)

    runs_root = config.get("runs_root", "runs")
    latest = find_latest_artifact_dir(runs_root, "inverse_kan")
    if latest and os.path.isdir(latest):
        return normpath(latest)

    if os.path.isdir(CONFIG_DIR):
        return normpath(CONFIG_DIR)

    return None


def forward_paths_from_dir(artifact_dir):
    if artifact_dir is None:
        return None, None
    return (
        normpath(os.path.join(artifact_dir, "kan_forward.pth")),
        normpath(os.path.join(artifact_dir, "kan_forward_meta.json")),
    )


def inverse_paths_from_dir(artifact_dir):
    if artifact_dir is None:
        return None, None
    return (
        normpath(os.path.join(artifact_dir, "kan_inverse.pth")),
        normpath(os.path.join(artifact_dir, "kan_inverse_meta.json")),
    )


def create_run_artifact_dir(runs_root, entry_name):
    run_dir = os.path.join(runs_root, f"{now_timestamp()}_{entry_name}")
    art_dir = os.path.join(run_dir, "artifacts")
    ensure_dir(art_dir)
    return normpath(run_dir), normpath(art_dir)


# -----------------------------------------------------------------------------
# 主系统
# -----------------------------------------------------------------------------
class FertilizerSystem:
    def __init__(self, data_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在初始化施肥控制系统 (Device: {self.device})...")
        ensure_dir(CONFIG_DIR)

        self.config = self._load_or_init_config(data_path=data_path)
        self.seed = int(self.config["seed"])
        set_seed(self.seed)
        self.data_path = self.config["data_path"]

        self.training_config = dict(DEFAULT_TRAINING_CONFIG)

        self.loaded_forward_artifact = False
        self.loaded_inverse_artifact = False

        self.forward_artifact_dir = None
        self.inverse_artifact_dir = None

        self.forward_weight_path = None
        self.forward_meta_path = None
        self.inverse_weight_path = None
        self.inverse_meta_path = None

        self._init_runtime_state()

        self.forward_model = self._get_forward_model()
        self.inverse_model = self._get_inverse_model()

        self._save_meta()
        print("\n>>> 系统就绪！当前系统配置已保存至 path/model_meta.json")

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

    def _save_json_atomic(self, path, payload):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_meta_",
            suffix=".json",
            dir=parent if parent else "."
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _save_meta(self):
        clean_config = {k: self.config.get(k) for k in DEFAULT_CONFIG.keys()}
        self._save_json_atomic(META_PATH, clean_config)

    def show_config(self):
        print("\n===== 当前交互配置 =====")
        for k, v in self.config.items():
            print(f"{k}: {v}")

        print("\n===== 当前运行状态 =====")
        print(f"forward_artifact_loaded: {self.loaded_forward_artifact}")
        print(f"inverse_artifact_loaded: {self.loaded_inverse_artifact}")
        print(f"forward_artifact_dir: {self.forward_artifact_dir}")
        print(f"inverse_artifact_dir: {self.inverse_artifact_dir}")
        print(f"forward_weight_path: {self.forward_weight_path}")
        print(f"forward_meta_path: {self.forward_meta_path}")
        print(f"inverse_weight_path: {self.inverse_weight_path}")
        print(f"inverse_meta_path: {self.inverse_meta_path}")
        print(f"forward_weight_exists: {safe_exists(self.forward_weight_path)}")
        print(f"inverse_weight_exists: {safe_exists(self.inverse_weight_path)}")
        print(f"forward_meta_exists: {safe_exists(self.forward_meta_path)}")
        print(f"inverse_meta_exists: {safe_exists(self.inverse_meta_path)}")

        print("\n===== 当前重训练参数 =====")
        for k, v in self.training_config.items():
            print(f"{k}: {v}")

    # =========================
    # 启动阶段：严格加载工件
    # =========================
    def _init_runtime_state(self):
        prefer_artifacts = bool(self.config.get("prefer_saved_artifacts", True))
        strict_mode = bool(self.config.get("strict_artifact_mode", True))

        if not prefer_artifacts:
            if strict_mode:
                raise RuntimeError(
                    "当前配置不允许跳过工件加载（strict_artifact_mode=True），"
                    "请开启 prefer_saved_artifacts 或关闭严格模式。"
                )
            raise RuntimeError("当前版本不支持在关闭工件优先时启动。")

        fwd_ok = self._try_load_forward_artifact()
        inv_ok = self._try_load_inverse_artifact()

        if fwd_ok and inv_ok:
            print("已从 forward / inverse 工件恢复归一化参数与训练域边界。")
            return

        if strict_mode:
            raise RuntimeError(
                "工件恢复失败：系统要求 forward / inverse 工件完整可用，"
                "为避免部署模型与论文模型漂移，已禁止启动阶段自动重训练。"
            )

        raise RuntimeError("当前版本不支持在工件缺失时继续启动。")

    def _try_load_forward_artifact(self):
        self.forward_artifact_dir = resolve_forward_artifact_dir(self.config)
        self.forward_weight_path, self.forward_meta_path = forward_paths_from_dir(self.forward_artifact_dir)

        if not safe_exists(self.forward_meta_path):
            print(f"缺少正向元数据文件: {self.forward_meta_path}")
            self.loaded_forward_artifact = False
            return False

        try:
            meta = load_json(self.forward_meta_path)

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

            self.forward_hidden_dim = int(hyper.get("hidden_dim", 8))

            self.loaded_forward_artifact = True
            self.config["forward_artifact_dir"] = self.forward_artifact_dir
            print(f"已读取正向工件: {self.forward_meta_path}")
            return True

        except Exception as e:
            print(f"读取正向工件失败（{e}）。")
            self.loaded_forward_artifact = False
            return False

    def _try_load_inverse_artifact(self):
        self.inverse_artifact_dir = resolve_inverse_artifact_dir(self.config)
        self.inverse_weight_path, self.inverse_meta_path = inverse_paths_from_dir(self.inverse_artifact_dir)

        if not safe_exists(self.inverse_meta_path):
            print(f"缺少反向元数据文件: {self.inverse_meta_path}")
            self.loaded_inverse_artifact = False
            return False

        try:
            meta = load_json(self.inverse_meta_path)

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

            self.inverse_hidden_dim = int(hyper.get("hidden_dim", 16))

            self.loaded_inverse_artifact = True
            self.config["inverse_artifact_dir"] = self.inverse_artifact_dir
            print(f"已读取反向工件: {self.inverse_meta_path}")
            return True

        except Exception as e:
            print(f"读取反向工件失败（{e}）。")
            self.loaded_inverse_artifact = False
            return False

    # =========================
    # 工具函数
    # =========================
    def _norm_forward_x(self, x):
        return (x - self.X_min) / (self.X_max - self.X_min + 1e-8)

    def _denorm_forward_y(self, y_norm):
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min

    def _norm_inverse_x(self, x):
        return (x - self.inv_x_min) / (self.inv_x_max - self.inv_x_min + 1e-8)

    def _denorm_inverse_y(self, y_norm):
        return y_norm * (self.inv_y_max - self.inv_y_min + 1e-8) + self.inv_y_min

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
    # 模型构建 / 加载
    # =========================
    def _build_forward_model(self):
        return FertilizerKAN(
            input_dim=2,
            hidden_dim=int(self.forward_hidden_dim),
            output_dim=1
        ).to(self.device)

    def _build_inverse_model(self):
        return InverseKANModel(
            input_dim=2,
            hidden_dim=int(self.inverse_hidden_dim),
            output_dim=1
        ).to(self.device)

    def _get_forward_model(self):
        if not safe_exists(self.forward_weight_path):
            raise FileNotFoundError(
                f"缺少正向权重文件: {self.forward_weight_path}。"
                "系统已禁止启动阶段自动重训练，请先通过训练脚本生成工件，"
                "或在菜单中显式选择重新训练。"
            )

        model = self._build_forward_model()
        print(f"[- 正向模型] 从 {self.forward_weight_path} 加载...")
        try:
            model.load_state_dict(torch.load(self.forward_weight_path, map_location=self.device))
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(
                f"正向模型权重加载失败: {e}。"
                "请检查权重文件与 meta 中记录的 hidden_dim 是否一致。"
            )

    def _get_inverse_model(self):
        if not safe_exists(self.inverse_weight_path):
            raise FileNotFoundError(
                f"缺少反向权重文件: {self.inverse_weight_path}。"
                "系统已禁止启动阶段自动重训练，请先通过训练脚本生成工件，"
                "或在菜单中显式选择重新训练。"
            )

        model = self._build_inverse_model()
        print(f"[- 反向模型] 从 {self.inverse_weight_path} 加载...")
        try:
            model.load_state_dict(torch.load(self.inverse_weight_path, map_location=self.device))
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(
                f"反向模型权重加载失败: {e}。"
                "请检查权重文件与 meta 中记录的 hidden_dim 是否一致。"
            )

    # =========================
    # 精确对接正式训练脚本
    # =========================
    def _retrain_forward_via_official_pipeline(self):
        print("\n>>> 调用正式正向训练入口: train_and_eval_kan(...)")

        runs_root = self.config.get("runs_root", "runs")
        _, artifact_dir = create_run_artifact_dir(runs_root, "train_kan")

        result = train_and_eval_kan(
            data_path=self.data_path,
            hidden_dim_candidates=self.training_config["forward_hidden_dim_candidates"],
            lr_candidates=self.training_config["forward_lr_candidates"],
            weight_decay_candidates=self.training_config["forward_weight_decay_candidates"],
            epochs=int(self.training_config["forward_epochs"]),
            search_epochs=int(self.training_config["forward_search_epochs"]),
            gamma=float(self.training_config["forward_lr_gamma"]),
            seed=int(self.seed),
            save_artifacts=True,
            artifact_dir=artifact_dir,
            weight_filename="kan_forward.pth",
            meta_filename="kan_forward_meta.json",
        )

        weight_path = os.path.join(artifact_dir, "kan_forward.pth")
        meta_path = os.path.join(artifact_dir, "kan_forward_meta.json")

        if not (os.path.exists(weight_path) and os.path.exists(meta_path)):
            raise RuntimeError("正式正向训练已返回，但未生成完整工件。")

        self.config["forward_artifact_dir"] = normpath(artifact_dir)
        self._save_meta()

        print(f"正向工件目录: {artifact_dir}")
        print(f"正向重训练完成: R²={result['r2']:.6f}, ARE={result['are']:.6f}%")
        return result

    def _retrain_inverse_via_official_pipeline(self):
        print("\n>>> 调用正式反向训练入口: train_and_eval_inverse_kan_v2(...)")

        runs_root = self.config.get("runs_root", "runs")
        _, artifact_dir = create_run_artifact_dir(runs_root, "inverse_kan")

        result = train_and_eval_inverse_kan_v2(
            data_path=self.data_path,
            hidden_dim_candidates=self.training_config["inverse_hidden_dim_candidates"],
            lr_candidates=self.training_config["inverse_lr_candidates"],
            weight_decay_candidates=self.training_config["inverse_weight_decay_candidates"],
            epochs=int(self.training_config["inverse_epochs"]),
            device=str(self.device),
            seed=int(self.seed),
            save_artifacts=True,
            artifact_dir=artifact_dir,
            weight_filename="kan_inverse.pth",
            meta_filename="kan_inverse_meta.json",
        )

        weight_path = os.path.join(artifact_dir, "kan_inverse.pth")
        meta_path = os.path.join(artifact_dir, "kan_inverse_meta.json")

        if not (os.path.exists(weight_path) and os.path.exists(meta_path)):
            raise RuntimeError("正式反向训练已返回，但未生成完整工件。")

        self.config["inverse_artifact_dir"] = normpath(artifact_dir)
        self._save_meta()

        print(f"反向工件目录: {artifact_dir}")
        print(
            "反向重训练完成: "
            f"main R²={result['r2_main']}, main ARE={result['are_main']}%, "
            f"all R²={result['r2_all']}, all ARE={result['are_all']}%"
        )
        return result

    def retrain_all(self):
        print("\n>>> 开始重新训练所有模型（显式操作，严格复用正式训练脚本）...")

        self._retrain_forward_via_official_pipeline()
        self._retrain_inverse_via_official_pipeline()

        # 重训练完成后，重新读取 meta / 权重，确保交互端状态与工件完全一致
        if not self._try_load_forward_artifact():
            raise RuntimeError("正向工件重训练完成后重新加载失败。")
        if not self._try_load_inverse_artifact():
            raise RuntimeError("反向工件重训练完成后重新加载失败。")

        self.forward_model = self._get_forward_model()
        self.inverse_model = self._get_inverse_model()

        self._save_meta()
        print(">>> 所有模型已按正式训练脚本重建并重新加载。")

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

        raw_pred_mass = float(self._denorm_forward_y(pred_norm))

        clipped = False
        final_pred_mass = raw_pred_mass
        if self.config.get("enable_output_clipping", True):
            final_pred_mass, clipped = self._clip_mass_to_physical_range(raw_pred_mass)

        return {
            "raw_pred_mass": float(raw_pred_mass),
            "pred_mass": float(final_pred_mass),
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

        raw_pred_speed = float(self._denorm_inverse_y(pred_speed_norm))

        clipped = False
        final_pred_speed = raw_pred_speed
        if self.config.get("enable_output_clipping", True):
            final_pred_speed, clipped = self._clip_speed_to_physical_range(raw_pred_speed)

        return {
            "opening": rec_opening,
            "raw_speed": float(raw_pred_speed),
            "speed": float(final_pred_speed),
            "warnings": warnings,
            "clipped": clipped
        }


def main():
    try:
        sys_ctrl = FertilizerSystem()
    except Exception as e:
        print(f"初始化错误: {e}")
        print(
            "提示：请先确保 forward / inverse 工件完整可用。"
            "推荐位置为 runs/<timestamp>_train_kan/artifacts/ 与 "
            "runs/<timestamp>_inverse_kan/artifacts/。"
        )
        return

    while True:
        print("\n" + "=" * 56)
        print("   排肥控制系统 v2.8 (Runs Artifact Compatible)")
        print("=" * 56)
        print("1. [正向预测] 开度/转速 -> 产量")
        print("2. [智能控制] 目标产量 -> 开度/转速")
        print("3. [系统维护] 重新训练模型（显式操作）")
        print("4. [系统维护] 查看当前配置")
        print("5. 退出")

        choice = input("请输入选项: ").strip()

        if choice == "1":
            try:
                op = float(input("请输入开度 (mm): "))
                spd = float(input("请输入转速 (r/min): "))

                result = sys_ctrl.predict_mass(op, spd)

                print(f"\n✅ 原始预测排肥量: {result['raw_pred_mass']:.2f} g/min")
                if result["clipped"]:
                    print(f"✅ 裁剪后输出排肥量: {result['pred_mass']:.2f} g/min")
                else:
                    print(f"✅ 最终输出排肥量: {result['pred_mass']:.2f} g/min")

                if result["warnings"]:
                    print("⚠ 输入超出训练范围，当前结果属于外推预测，可信度可能下降：")
                    for msg in result["warnings"]:
                        print(f"   - {msg}")

                if result["clipped"]:
                    print("⚠ 预测结果已裁剪到训练数据支持的物理范围内。")
                    print("⚠ 请注意区分：原始预测值反映模型输出，裁剪后结果属于后处理结果。")

            except ValueError:
                print("输入错误，请输入数值。")

        elif choice == "2":
            try:
                target = float(input("请输入目标产量 (g/min): "))

                result = sys_ctrl.intelligent_control(target)

                print("\n✅ 推荐策略:")
                print(f"   - 调节开度: {result['opening']:.1f} mm")

                if result["clipped"]:
                    print(f"   - 原始推荐转速: {result['raw_speed']:.2f} r/min")
                    print(f"   - 裁剪后设定转速: {result['speed']:.2f} r/min")
                else:
                    print(f"   - 设定转速: {result['speed']:.2f} r/min")

                if result["warnings"]:
                    print("⚠ 输入超出训练范围，当前结果属于外推控制建议，可信度可能下降：")
                    for msg in result["warnings"]:
                        print(f"   - {msg}")

                if result["clipped"]:
                    print("⚠ 推荐转速已裁剪到训练数据支持的物理范围内。")
                    print("⚠ 请注意区分：原始转速反映模型输出，裁剪后结果属于后处理结果。")

            except ValueError:
                print("输入错误，请输入数值。")

        elif choice == "3":
            confirm = input(
                "确定要重新训练吗？这会基于当前数据重新生成模型，可能导致与既有论文工件不一致。(y/n): "
            ).strip().lower()
            if confirm == "y":
                try:
                    sys_ctrl.retrain_all()
                except Exception as e:
                    print(f"重新训练失败: {e}")

        elif choice == "4":
            sys_ctrl.show_config()

        elif choice == "5":
            print("系统已退出。")
            break

        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()