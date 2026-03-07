# -*- coding: utf-8 -*-
"""
inverse_kan.py

反向 KAN 模型（严格披露版）：
    输入:  [目标质量 (g/min), 排肥口开度 (mm)]
    输出:  [排肥轴转速 (r/min)]

任务定义（与论文口径一致）：
    本模型服务于“转速优先”控制方法：
        1) 先根据目标质量确定策略开度
        2) 在策略开度确定后，再调节排肥轴转速

因此：
    - 主评估对象：满足“实际开度 == 策略开度”的测试样本（策略一致子集）
    - 补充评估对象：全测试集（仅作透明展示，不作为主结论）

本版修订重点：
1. 保留原有训练与保存工件流程
2. 显式报告：
   - 主结果样本数 n_main
   - 全测试集样本数 n_all
   - 主结果占比 main_ratio
   - 主结果与全测试集的 R² / ARE 并列输出
3. 增加策略一致子集在不同开度上的样本分布
4. 对极小样本时的 R² / ARE 做安全处理
5. 返回更完整的结果字典，便于 compare / 论文表格 / 附录复用
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, average_relative_error


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0
EPS = 1e-8

DEFAULT_ARTIFACT_DIR = "path"
DEFAULT_WEIGHT_FILENAME = "kan_inverse.pth"
DEFAULT_META_FILENAME = "kan_inverse_meta.json"


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


def select_optimal_opening(target_mass: float) -> float:
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    elif target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    else:
        return 50.0


def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]):
        return np.nan
    return float(r2_score(y_true, y_pred))


def _safe_are(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(average_relative_error(y_true, y_pred))


def _count_openings(openings, opening_values=(20.0, 35.0, 50.0), atol=0.1):
    openings = np.asarray(openings, dtype=float)
    stats = {}
    for v in opening_values:
        stats[f"{int(v)}mm"] = int(np.isclose(openings, v, atol=atol).sum())
    stats["other"] = int(
        len(openings)
        - sum(stats[k] for k in ["20mm", "35mm", "50mm"])
    )
    return stats


class KANLayer(nn.Module):
    """
    说明：
    当前实现保留作者原始结构，不在这一步修改模型名实一致性问题。
    若后续要进一步严格审稿整改，可单独把“反向 KAN”重命名为更准确的结构名称。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.base = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.linear(x) + self.base(x)


class InverseKANModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            KANLayer(input_dim, hidden_dim),
            nn.Tanh(),
            KANLayer(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def _ensure_save_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)


def _to_list(arr):
    return np.asarray(arr).tolist()


def save_inverse_artifacts(
    model,
    model_path,
    meta_path,
    *,
    seed,
    data_path,
    hidden_dim,
    lr,
    weight_decay,
    epochs,
    best_val_r2,
    x_min,
    x_max,
    y_min,
    y_max,
    x_train_full_raw,
    y_train_full_raw,
    idx_tr,
    idx_val,
    idx_te,
):
    torch.save(model.state_dict(), model_path)

    meta = {
        "artifact_type": "inverse_model_bundle",
        "model_name": "KAN",
        "model_class": "InverseKANModel",
        "weight_path": model_path.replace("\\", "/"),
        "data_path": data_path,
        "seed": int(seed),
        "task_definition": {
            "input": ["target_mass_g_min", "opening_mm"],
            "output": "speed_r_min",
            "policy": "speed_first",
            "strategy_opening_rule": {
                "threshold_low_mid": float(THRESHOLD_LOW_MID),
                "threshold_mid_high": float(THRESHOLD_MID_HIGH),
                "opening_low": 20.0,
                "opening_mid": 35.0,
                "opening_high": 50.0,
            }
        },
        "normalization_scope": {
            "inverse": "train+val"
        },
        "hyperparameters": {
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
        },
        "validation_result": {
            "best_val_r2": float(best_val_r2)
        },
        "normalization_params": {
            "X_min": _to_list(x_min),
            "X_max": _to_list(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "training_domain": {
            "target_mass_min": float(x_train_full_raw[:, 0].min()),
            "target_mass_max": float(x_train_full_raw[:, 0].max()),
            "opening_min": float(x_train_full_raw[:, 1].min()),
            "opening_max": float(x_train_full_raw[:, 1].max()),
            "speed_min": float(y_train_full_raw.min()),
            "speed_max": float(y_train_full_raw.max()),
        },
        "split_info": {
            "train_size": int(len(idx_tr)),
            "val_size": int(len(idx_val)),
            "test_size": int(len(idx_te)),
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"反向模型权重已保存：{model_path}")
    print(f"反向模型元数据已保存：{meta_path}")


def fit_one_inverse_kan(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dim=16,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=1000,
    device="cpu",
    seed=42,
):
    set_seed(seed)

    model = InverseKANModel(input_dim=2, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_state = None
    best_val_r2 = -np.inf

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy().reshape(-1)

        val_r2 = r2_score(y_val, val_pred)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_r2


def train_final_inverse_kan(
    X_train,
    y_train,
    hidden_dim=16,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=1000,
    device="cpu",
    seed=42,
):
    set_seed(seed)

    model = InverseKANModel(input_dim=2, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    return model


def train_and_eval_inverse_kan_v2(
    data_path="data/dataset.xlsx",
    hidden_dim_candidates=None,
    lr_candidates=None,
    weight_decay_candidates=None,
    epochs=1000,
    device=None,
    seed=42,
    save_artifacts=True,
    artifact_dir=DEFAULT_ARTIFACT_DIR,
    weight_filename=DEFAULT_WEIGHT_FILENAME,
    meta_filename=DEFAULT_META_FILENAME,
):
    if hidden_dim_candidates is None:
        hidden_dim_candidates = [8, 16, 32]
    if lr_candidates is None:
        lr_candidates = [1e-2, 5e-3, 1e-3]
    if weight_decay_candidates is None:
        weight_decay_candidates = [0.0, 1e-6, 1e-5, 1e-4]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== 训练 反向 KAN（严格披露版，转速优先口径） ===")
    print(f"使用设备: {device}")

    X_raw, y_raw = load_data(data_path)

    n_samples = len(y_raw)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X_raw, y=y_raw)

    # 反向任务：
    # 输入 = [目标质量, 开度]
    # 输出 = 转速
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]
    y_test_raw = y_inv_all[idx_te]
    _ = y_test_raw  # 保留变量语义，避免静态检查提示

    # 与作者当前口径一致：反向归一化统计量使用 train+val
    train_full_idx = np.concatenate([idx_tr, idx_val])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    x_min = X_train_full_raw.min(axis=0, keepdims=True)
    x_max = X_train_full_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_full_raw.min())
    y_max = float(y_train_full_raw.max())

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + EPS)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + EPS)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + EPS) + y_min

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    X_test = norm_x(X_test_raw)

    y_train = norm_y(y_train_raw)
    y_val = norm_y(y_val_raw)

    best_val_r2 = -np.inf
    best_hidden_dim = None
    best_lr = None
    best_weight_decay = None

    print(">>> 开始搜索反向 KAN 最优超参数（基于 val R²，归一化空间）...")
    for hidden_dim in hidden_dim_candidates:
        for lr in lr_candidates:
            for wd in weight_decay_candidates:
                _, val_r2 = fit_one_inverse_kan(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    weight_decay=wd,
                    epochs=epochs,
                    device=device,
                    seed=seed,
                )

                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_hidden_dim = int(hidden_dim)
                    best_lr = float(lr)
                    best_weight_decay = float(wd)

    if best_hidden_dim is None or best_lr is None or best_weight_decay is None:
        raise RuntimeError("反向 KAN 超参数搜索失败：最优参数为空")

    print(
        f"反向 KAN 最优超参数：hidden_dim={best_hidden_dim}, "
        f"lr={best_lr}, weight_decay={best_weight_decay}, val R²={best_val_r2:.6f}"
    )

    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])

    X_train_val = norm_x(X_train_val_raw)
    y_train_val = norm_y(y_train_val_raw)

    model_final = train_final_inverse_kan(
        X_train=X_train_val,
        y_train=y_train_val,
        hidden_dim=best_hidden_dim,
        lr=best_lr,
        weight_decay=best_weight_decay,
        epochs=epochs,
        device=device,
        seed=seed,
    )

    print("\n>>> 开始测试集评估（主结果=策略一致子集，补充=全测试集）...")

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    model_final.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred_test_norm = model_final(X_test_t).cpu().numpy().reshape(-1)

    pred_test_speed = denorm_y(pred_test_norm)

    # 全测试集指标
    r2_all = _safe_r2(test_speed_true, pred_test_speed)
    are_all = _safe_are(test_speed_true, pred_test_speed)
    n_all = int(len(test_speed_true))

    # 策略一致子集
    strategy_opening_all = np.array(
        [select_optimal_opening(float(m)) for m in test_mass],
        dtype=float
    )
    policy_mask = np.isclose(test_opening, strategy_opening_all, atol=0.1)

    n_main = int(policy_mask.sum())
    main_ratio = float(n_main / n_all) if n_all > 0 else np.nan

    if n_main > 0:
        mass_main = test_mass[policy_mask]
        opening_main = test_opening[policy_mask]
        strategy_opening_main = strategy_opening_all[policy_mask]
        y_true_main = test_speed_true[policy_mask]
        y_pred_main = pred_test_speed[policy_mask]

        r2_main = _safe_r2(y_true_main, y_pred_main)
        are_main = _safe_are(y_true_main, y_pred_main)
    else:
        mass_main = np.array([], dtype=float)
        opening_main = np.array([], dtype=float)
        strategy_opening_main = np.array([], dtype=float)
        y_true_main = np.array([], dtype=float)
        y_pred_main = np.array([], dtype=float)
        r2_main = np.nan
        are_main = np.nan

    # 分布披露：全测试集 / 主结果子集的开度分布
    opening_dist_all = _count_openings(test_opening)
    opening_dist_main = _count_openings(opening_main)

    print("\n===== 反向 KAN（转速优先口径）测试结果 =====")
    if n_main > 0:
        print(
            f"主结果（策略一致子集）: n = {n_main:3d} / {n_all:3d} "
            f"({main_ratio * 100:.2f}%), R² = {r2_main:.4f}, ARE = {are_main:.4f}%"
        )
    else:
        print(
            f"主结果（策略一致子集）: n =   0 / {n_all:3d} "
            f"(0.00%), R² = NaN, ARE = NaN"
        )

    print(
        f"补充结果（全测试集）  : n = {n_all:3d}, "
        f"R² = {r2_all:.4f}, ARE = {are_all:.4f}%"
    )

    print("\n--- 开度分布披露 ---")
    print(
        "全测试集开度分布: "
        f"20mm={opening_dist_all['20mm']}, "
        f"35mm={opening_dist_all['35mm']}, "
        f"50mm={opening_dist_all['50mm']}, "
        f"other={opening_dist_all['other']}"
    )
    print(
        "主结果子集开度分布: "
        f"20mm={opening_dist_main['20mm']}, "
        f"35mm={opening_dist_main['35mm']}, "
        f"50mm={opening_dist_main['50mm']}, "
        f"other={opening_dist_main['other']}"
    )

    print("\n--- 结果解释建议 ---")
    print("主结果基于‘实际开度 = 策略推荐开度’的测试样本。")
    print("为避免选择性报告，应始终与全测试集结果并列呈现，并说明主结果样本占比。")

    model_path = None
    meta_path = None
    if save_artifacts:
        _ensure_save_dir(artifact_dir)
        model_path = os.path.join(artifact_dir, weight_filename)
        meta_path = os.path.join(artifact_dir, meta_filename)

        save_inverse_artifacts(
            model_final,
            model_path=model_path,
            meta_path=meta_path,
            seed=seed,
            data_path=data_path,
            hidden_dim=best_hidden_dim,
            lr=best_lr,
            weight_decay=best_weight_decay,
            epochs=epochs,
            best_val_r2=best_val_r2,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_train_full_raw=X_train_full_raw,
            y_train_full_raw=y_train_full_raw,
            idx_tr=idx_tr,
            idx_val=idx_val,
            idx_te=idx_te,
        )

    return {
        # 主结果
        "r2_main": float(r2_main) if not np.isnan(r2_main) else np.nan,
        "are_main": float(are_main) if not np.isnan(are_main) else np.nan,
        "n_main": n_main,
        "main_ratio": float(main_ratio) if not np.isnan(main_ratio) else np.nan,

        # 全测试集
        "r2_all": float(r2_all) if not np.isnan(r2_all) else np.nan,
        "are_all": float(are_all) if not np.isnan(are_all) else np.nan,
        "n_all": n_all,

        # 最优超参数
        "best_hidden_dim": int(best_hidden_dim),
        "best_lr": float(best_lr),
        "best_weight_decay": float(best_weight_decay),

        # 全测试集明细
        "y_true_all": np.asarray(test_speed_true),
        "y_pred_all": np.asarray(pred_test_speed),
        "mass_all": np.asarray(test_mass),
        "opening_all": np.asarray(test_opening),
        "strategy_opening_all": np.asarray(strategy_opening_all),

        # 主结果子集明细
        "y_true_main": np.asarray(y_true_main),
        "y_pred_main": np.asarray(y_pred_main),
        "mass_main": np.asarray(mass_main),
        "opening_main": np.asarray(opening_main),
        "strategy_opening_main": np.asarray(strategy_opening_main),

        # 策略一致掩码
        "policy_mask": np.asarray(policy_mask),

        # 分布披露
        "opening_dist_all": opening_dist_all,
        "opening_dist_main": opening_dist_main,

        # 工件信息
        "artifact_model_path": model_path,
        "artifact_meta_path": meta_path,
    }


if __name__ == "__main__":
    train_and_eval_inverse_kan_v2()