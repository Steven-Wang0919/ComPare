# -*- coding: utf-8 -*-
"""
evaluate_inverse_opening_holdout.py

补充实验：
- 固定原始策略阈值不变：<2800 -> 20 mm, 2800~4800 -> 35 mm, >=4800 -> 50 mm
- 将所有 actual_opening in {20, 35, 50} 的样本整体留作反向任务测试集
- 剩余非 {20, 35, 50} 样本用于 train / val
- 比较 inverse_MLP / inverse_GRNN / inverse_KAN 在该“策略开度整档留出”设定下的表现

输出：
- inverse_opening_holdout_metrics.csv
- inverse_opening_holdout_predictions_all.csv
- inverse_opening_holdout_predictions_main.csv
- run_manifest.json
"""

import os
import random
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from common_utils import load_data, average_relative_error
from run_utils import (
    append_manifest_outputs,
    create_run_dir,
    ensure_dir,
    save_dataframe,
    write_manifest,
)


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0
TARGET_OPENINGS = (20.0, 35.0, 50.0)
OPENING_ATOL = 0.1
EPS = 1e-8


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
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]):
        return np.nan
    return float(r2_score(y_true, y_pred))


def _safe_are(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) == 0:
        return np.nan
    return float(average_relative_error(y_true, y_pred))


def _count_openings(openings, opening_values=(20.0, 35.0, 50.0), atol=0.1):
    openings = np.asarray(openings, dtype=float).reshape(-1)
    stats = {}
    for v in opening_values:
        stats[f"{int(v)}mm"] = int(np.isclose(openings, v, atol=atol).sum())
    stats["other"] = int(len(openings) - sum(stats[k] for k in ["20mm", "35mm", "50mm"]))
    return stats


def _to_1d_array(arr, name):
    out = np.asarray(arr).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} 为空，无法继续。")
    return out


def _is_target_opening(openings, target_openings=TARGET_OPENINGS, atol=OPENING_ATOL):
    openings = np.asarray(openings, dtype=float).reshape(-1)
    mask = np.zeros(len(openings), dtype=bool)
    for op in target_openings:
        mask |= np.isclose(openings, op, atol=atol)
    return mask


def build_opening_holdout_indices(X, y, random_state=42, val_ratio=0.2):
    """
    按开度整档留出：
    - test: actual_opening in {20, 35, 50}
    - train/val: actual_opening not in {20, 35, 50}
    """
    opening = np.asarray(X[:, 0], dtype=float).reshape(-1)

    test_mask = _is_target_opening(opening)
    idx_te = np.where(test_mask)[0]
    idx_train_val = np.where(~test_mask)[0]

    if len(idx_te) == 0:
        raise ValueError("未找到开度为 20/35/50 mm 的样本，无法构造 holdout 测试集。")

    if len(idx_train_val) < 2:
        raise ValueError("非 20/35/50 mm 样本过少，无法构造训练/验证集。")

    idx_tr, idx_val = train_test_split(
        idx_train_val,
        test_size=val_ratio,
        shuffle=True,
        random_state=random_state,
    )

    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_val = np.asarray(idx_val, dtype=int)
    idx_te = np.asarray(idx_te, dtype=int)

    if len(idx_tr) == 0 or len(idx_val) == 0:
        raise ValueError("划分后 train 或 val 为空，请检查数据集分布。")

    return idx_tr, idx_val, idx_te


def _make_inverse_xy(X_raw, y_raw):
    """
    反向任务输入: [target_mass, opening]
    反向任务输出: speed
    """
    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]
    return X_inv_all, y_inv_all


def _build_norm_funcs(X_train_raw, y_train_raw):
    x_min = X_train_raw.min(axis=0, keepdims=True)
    x_max = X_train_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_raw.min())
    y_max = float(y_train_raw.max())

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + EPS)

    def norm_y(y):
        return (y - y_min) / (y_max - y_min + EPS)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + EPS) + y_min

    return norm_x, norm_y, denorm_y, x_min, x_max, y_min, y_max


# =========================
# inverse_MLP holdout
# =========================
def train_and_eval_inverse_mlp_holdout(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=42,
    save_outputs_dir=None,
):
    if hidden_layer_candidates is None:
        hidden_layer_candidates = [(10,), (20,), (50,), (20, 20)]
    if alpha_candidates is None:
        alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    print("\n=== 训练 反向 MLP（opening-holdout，train-only normalization） ===")

    X_raw, y_raw = load_data(data_path)
    idx_tr, idx_val, idx_te = build_opening_holdout_indices(X_raw, y_raw, random_state=random_state)

    X_inv_all, y_inv_all = _make_inverse_xy(X_raw, y_raw)

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]

    norm_x, norm_y, denorm_y, _, _, _, _ = _build_norm_funcs(X_train_raw, y_train_raw)

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    y_train_norm = norm_y(y_train_raw)

    best_r2_val = -np.inf
    best_hidden = None
    best_alpha = None

    for h in hidden_layer_candidates:
        for a in alpha_candidates:
            model = MLPRegressor(
                hidden_layer_sizes=h,
                alpha=a,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )
            model.fit(X_train, y_train_norm)

            y_val_pred_norm = model.predict(X_val)
            y_val_pred = denorm_y(y_val_pred_norm)
            r2_val = _safe_r2(y_val_raw, y_val_pred)

            if not np.isnan(r2_val) and r2_val > best_r2_val:
                best_r2_val = r2_val
                best_hidden = h
                best_alpha = a

    if best_hidden is None or best_alpha is None:
        raise RuntimeError("反向 MLP 超参数搜索失败：best_hidden / best_alpha 为空")

    print(
        f"反向 MLP 最优超参数：hidden_layer_sizes={best_hidden}, "
        f"alpha={best_alpha}, val R²={best_r2_val:.6f}"
    )

    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])

    X_train_val = norm_x(X_train_val_raw)
    y_train_val_norm = norm_y(y_train_val_raw)

    mlp_final = MLPRegressor(
        hidden_layer_sizes=best_hidden,
        alpha=best_alpha,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp_final.fit(X_train_val, y_train_val_norm)

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    X_test_norm = norm_x(X_test_raw)
    pred_test_norm = mlp_final.predict(X_test_norm)
    pred_test_speed = denorm_y(pred_test_norm)

    r2_all = _safe_r2(test_speed_true, pred_test_speed)
    are_all = _safe_are(test_speed_true, pred_test_speed)
    n_all = int(len(test_speed_true))

    strategy_opening_all = np.array(
        [select_optimal_opening(float(m)) for m in test_mass],
        dtype=float,
    )
    policy_mask = np.isclose(test_opening, strategy_opening_all, atol=OPENING_ATOL)

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

    opening_dist_all = _count_openings(test_opening)
    opening_dist_main = _count_openings(opening_main)

    if save_outputs_dir is not None:
        df_all = pd.DataFrame({
            "target_mass_g_min": test_mass,
            "actual_opening_mm": test_opening,
            "strategy_opening_mm": strategy_opening_all,
            "true_speed_r_min": test_speed_true,
            "inverse_MLP_pred": pred_test_speed,
            "policy_match": policy_mask.astype(int),
        })
        save_dataframe(df_all, os.path.join(save_outputs_dir, "inverse_mlp_predictions_all.csv"))

        df_main = pd.DataFrame({
            "target_mass_g_min": mass_main,
            "actual_opening_mm": opening_main,
            "strategy_opening_mm": strategy_opening_main,
            "true_speed_r_min": y_true_main,
            "inverse_MLP_pred": y_pred_main,
        })
        save_dataframe(df_main, os.path.join(save_outputs_dir, "inverse_mlp_predictions_main.csv"))

    return {
        "r2_main": r2_main,
        "are_main": are_main,
        "n_main": n_main,
        "main_ratio": main_ratio,
        "r2_all": r2_all,
        "are_all": are_all,
        "n_all": n_all,
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,
        "y_true_all": np.asarray(test_speed_true),
        "y_pred_all": np.asarray(pred_test_speed),
        "mass_all": np.asarray(test_mass),
        "opening_all": np.asarray(test_opening),
        "strategy_opening_all": np.asarray(strategy_opening_all),
        "y_true_main": np.asarray(y_true_main),
        "y_pred_main": np.asarray(y_pred_main),
        "mass_main": np.asarray(mass_main),
        "opening_main": np.asarray(opening_main),
        "strategy_opening_main": np.asarray(strategy_opening_main),
        "policy_mask": np.asarray(policy_mask),
        "opening_dist_all": opening_dist_all,
        "opening_dist_main": opening_dist_main,
    }


# =========================
# inverse_GRNN holdout
# =========================
class InverseGRNN:
    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        if self.X.ndim != 2:
            raise ValueError("X 必须是二维数组")
        if len(self.X) != len(self.y):
            raise ValueError("X 和 y 的样本数必须一致")
        if len(self.X) == 0:
            raise ValueError("训练数据不能为空")

    def _predict_one(self, x):
        diff = self.X - x
        dist2 = np.sum(diff ** 2, axis=1)
        w = np.exp(-dist2 / (2.0 * self.sigma ** 2))
        w_sum = w.sum()
        if w_sum <= EPS:
            return float(self.y.mean())
        return float(np.sum(w * self.y) / w_sum)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x) for x in X], dtype=float)


def train_and_eval_inverse_grnn_holdout(
    data_path="data/dataset.xlsx",
    sigma_grid=None,
    random_state=42,
    save_outputs_dir=None,
):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.1, 4.0, 40)

    sigma_grid = np.asarray(sigma_grid, dtype=float).reshape(-1)
    if len(sigma_grid) == 0:
        raise ValueError("sigma_grid 不能为空")

    print("\n=== 训练 反向 GRNN（opening-holdout，train-only normalization） ===")

    X_raw, y_raw = load_data(data_path)
    idx_tr, idx_val, idx_te = build_opening_holdout_indices(X_raw, y_raw, random_state=random_state)

    X_inv_all, y_inv_all = _make_inverse_xy(X_raw, y_raw)

    X_train_raw = X_inv_all[idx_tr]
    y_train_raw = y_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    y_val_raw = y_inv_all[idx_val]

    norm_x, norm_y, denorm_y, _, _, _, _ = _build_norm_funcs(X_train_raw, y_train_raw)

    X_train_norm = norm_x(X_train_raw)
    X_val_norm = norm_x(X_val_raw)
    y_train_norm = norm_y(y_train_raw)

    best_sigma = None
    best_r2_val = -np.inf

    for s in sigma_grid:
        model_tmp = InverseGRNN(sigma=float(s))
        model_tmp.fit(X_train_norm, y_train_norm)
        y_val_pred_norm = model_tmp.predict(X_val_norm)
        y_val_pred_raw = denorm_y(y_val_pred_norm)
        r2_val = _safe_r2(y_val_raw, y_val_pred_raw)

        if not np.isnan(r2_val) and r2_val > best_r2_val:
            best_r2_val = r2_val
            best_sigma = float(s)

    if best_sigma is None:
        raise RuntimeError("反向 GRNN 超参数搜索失败：best_sigma 为空")

    print(f"反向 GRNN 最优 sigma={best_sigma:.4f}, val R²={best_r2_val:.6f}")

    train_full_idx = np.concatenate([idx_tr, idx_val])
    X_train_full_raw = X_inv_all[train_full_idx]
    y_train_full_raw = y_inv_all[train_full_idx]

    X_train_full_norm = norm_x(X_train_full_raw)
    y_train_full_norm = norm_y(y_train_full_raw)

    inv_grnn = InverseGRNN(sigma=best_sigma)
    inv_grnn.fit(X_train_full_norm, y_train_full_norm)

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    X_test_raw = np.stack([test_mass, test_opening], axis=1)
    X_test_norm = norm_x(X_test_raw)

    pred_test_norm = inv_grnn.predict(X_test_norm)
    pred_test_speed = denorm_y(pred_test_norm)

    r2_all = _safe_r2(test_speed_true, pred_test_speed)
    are_all = _safe_are(test_speed_true, pred_test_speed)
    n_all = int(len(test_speed_true))

    strategy_opening_all = np.array(
        [select_optimal_opening(float(m)) for m in test_mass],
        dtype=float,
    )
    policy_mask = np.isclose(test_opening, strategy_opening_all, atol=OPENING_ATOL)

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

    opening_dist_all = _count_openings(test_opening)
    opening_dist_main = _count_openings(opening_main)

    if save_outputs_dir is not None:
        df_all = pd.DataFrame({
            "target_mass_g_min": test_mass,
            "actual_opening_mm": test_opening,
            "strategy_opening_mm": strategy_opening_all,
            "true_speed_r_min": test_speed_true,
            "inverse_GRNN_pred": pred_test_speed,
            "policy_match": policy_mask.astype(int),
        })
        save_dataframe(df_all, os.path.join(save_outputs_dir, "inverse_grnn_predictions_all.csv"))

        df_main = pd.DataFrame({
            "target_mass_g_min": mass_main,
            "actual_opening_mm": opening_main,
            "strategy_opening_mm": strategy_opening_main,
            "true_speed_r_min": y_true_main,
            "inverse_GRNN_pred": y_pred_main,
        })
        save_dataframe(df_main, os.path.join(save_outputs_dir, "inverse_grnn_predictions_main.csv"))

    return {
        "r2_main": r2_main,
        "are_main": are_main,
        "n_main": n_main,
        "main_ratio": main_ratio,
        "r2_all": r2_all,
        "are_all": are_all,
        "n_all": n_all,
        "best_sigma": best_sigma,
        "y_true_all": np.asarray(test_speed_true),
        "y_pred_all": np.asarray(pred_test_speed),
        "mass_all": np.asarray(test_mass),
        "opening_all": np.asarray(test_opening),
        "strategy_opening_all": np.asarray(strategy_opening_all),
        "y_true_main": np.asarray(y_true_main),
        "y_pred_main": np.asarray(y_pred_main),
        "mass_main": np.asarray(mass_main),
        "opening_main": np.asarray(opening_main),
        "strategy_opening_main": np.asarray(strategy_opening_main),
        "policy_mask": np.asarray(policy_mask),
        "opening_dist_all": opening_dist_all,
        "opening_dist_main": opening_dist_main,
    }


# =========================
# inverse_KAN holdout
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
        super().__init__()
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
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.input_grid
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
        A = self.b_splines(x.transpose(0, 1)).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor):
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out


class InverseKANModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=1):
        super().__init__()
        self.kan1 = KANLayer(input_dim, hidden_dim, grid_size=10)
        self.kan2 = KANLayer(hidden_dim, output_dim, grid_size=10)

    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        return x


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
    x_train_domain_raw,
    y_train_domain_raw,
    idx_tr,
    idx_val,
    idx_te,
):
    torch.save(model.state_dict(), model_path)

    meta = {
        "artifact_type": "inverse_model_bundle",
        "model_name": "KAN",
        "model_class": "InverseKANModel",
        "architecture_repair": True,
        "weight_path": model_path.replace("\\", "/"),
        "data_path": data_path,
        "seed": int(seed),
        "hyperparameters": {
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
        },
        "validation_result": {"best_val_r2": float(best_val_r2)},
        "normalization_params": {
            "X_min": _to_list(x_min),
            "X_max": _to_list(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "training_domain": {
            "target_mass_min": float(x_train_domain_raw[:, 0].min()),
            "target_mass_max": float(x_train_domain_raw[:, 0].max()),
            "opening_min": float(x_train_domain_raw[:, 1].min()),
            "opening_max": float(x_train_domain_raw[:, 1].max()),
            "speed_min": float(y_train_domain_raw.min()),
            "speed_max": float(y_train_domain_raw.max()),
        },
        "split_info": {
            "train_size": int(len(idx_tr)),
            "val_size": int(len(idx_val)),
            "test_size": int(len(idx_te)),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def fit_one_inverse_kan(
    X_train,
    y_train,
    X_val,
    y_val_raw,
    denorm_y,
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
            val_pred_norm = model(X_val_t).cpu().numpy().reshape(-1)

        val_pred_raw = denorm_y(val_pred_norm)
        val_r2 = _safe_r2(y_val_raw, val_pred_raw)

        if not np.isnan(val_r2) and val_r2 > best_val_r2:
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


def train_and_eval_inverse_kan_holdout(
    data_path="data/dataset.xlsx",
    hidden_dim_candidates=None,
    lr_candidates=None,
    weight_decay_candidates=None,
    epochs=1000,
    device=None,
    seed=42,
    save_artifacts=False,
    artifact_dir=None,
    weight_filename="kan_inverse_holdout.pth",
    meta_filename="kan_inverse_holdout_meta.json",
    save_outputs_dir=None,
):
    if hidden_dim_candidates is None:
        hidden_dim_candidates = [8, 16, 32]
    if lr_candidates is None:
        lr_candidates = [1e-2, 5e-3, 1e-3]
    if weight_decay_candidates is None:
        weight_decay_candidates = [0.0, 1e-6, 1e-5, 1e-4]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== 训练 反向 KAN（opening-holdout，train-only normalization） ===")
    print(f"使用设备: {device}")

    X_raw, y_raw = load_data(data_path)
    idx_tr, idx_val, idx_te = build_opening_holdout_indices(X_raw, y_raw, random_state=seed)

    X_inv_all, y_inv_all = _make_inverse_xy(X_raw, y_raw)

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]

    norm_x, norm_y, denorm_y, x_min, x_max, y_min, y_max = _build_norm_funcs(X_train_raw, y_train_raw)

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    X_test = norm_x(X_test_raw)
    y_train = norm_y(y_train_raw)

    best_val_r2 = -np.inf
    best_hidden_dim = None
    best_lr = None
    best_weight_decay = None

    for hidden_dim in hidden_dim_candidates:
        for lr in lr_candidates:
            for wd in weight_decay_candidates:
                _, val_r2 = fit_one_inverse_kan(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val_raw=y_val_raw,
                    denorm_y=denorm_y,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    weight_decay=wd,
                    epochs=epochs,
                    device=device,
                    seed=seed,
                )

                if not np.isnan(val_r2) and val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_hidden_dim = int(hidden_dim)
                    best_lr = float(lr)
                    best_weight_decay = float(wd)

    if best_hidden_dim is None:
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

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    model_final.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred_test_norm = model_final(X_test_t).cpu().numpy().reshape(-1)

    pred_test_speed = denorm_y(pred_test_norm)

    r2_all = _safe_r2(test_speed_true, pred_test_speed)
    are_all = _safe_are(test_speed_true, pred_test_speed)
    n_all = int(len(test_speed_true))

    strategy_opening_all = np.array(
        [select_optimal_opening(float(m)) for m in test_mass],
        dtype=float,
    )
    policy_mask = np.isclose(test_opening, strategy_opening_all, atol=OPENING_ATOL)

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

    opening_dist_all = _count_openings(test_opening)
    opening_dist_main = _count_openings(opening_main)

    if save_outputs_dir is not None:
        df_all = pd.DataFrame({
            "target_mass_g_min": test_mass,
            "actual_opening_mm": test_opening,
            "strategy_opening_mm": strategy_opening_all,
            "true_speed_r_min": test_speed_true,
            "inverse_KAN_pred": pred_test_speed,
            "policy_match": policy_mask.astype(int),
        })
        save_dataframe(df_all, os.path.join(save_outputs_dir, "inverse_kan_predictions_all.csv"))

        df_main = pd.DataFrame({
            "target_mass_g_min": mass_main,
            "actual_opening_mm": opening_main,
            "strategy_opening_mm": strategy_opening_main,
            "true_speed_r_min": y_true_main,
            "inverse_KAN_pred": y_pred_main,
        })
        save_dataframe(df_main, os.path.join(save_outputs_dir, "inverse_kan_predictions_main.csv"))

    model_path = None
    meta_path = None
    if save_artifacts:
        if artifact_dir is None:
            raise ValueError("save_artifacts=True 时必须提供 artifact_dir")
        ensure_dir(artifact_dir)
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
            x_train_domain_raw=X_train_raw,
            y_train_domain_raw=y_train_raw,
            idx_tr=idx_tr,
            idx_val=idx_val,
            idx_te=idx_te,
        )

    return {
        "r2_main": float(r2_main) if not np.isnan(r2_main) else np.nan,
        "are_main": float(are_main) if not np.isnan(are_main) else np.nan,
        "n_main": n_main,
        "main_ratio": float(main_ratio) if not np.isnan(main_ratio) else np.nan,
        "r2_all": float(r2_all) if not np.isnan(r2_all) else np.nan,
        "are_all": float(are_all) if not np.isnan(are_all) else np.nan,
        "n_all": n_all,
        "best_hidden_dim": int(best_hidden_dim),
        "best_lr": float(best_lr),
        "best_weight_decay": float(best_weight_decay),
        "y_true_all": np.asarray(test_speed_true),
        "y_pred_all": np.asarray(pred_test_speed),
        "mass_all": np.asarray(test_mass),
        "opening_all": np.asarray(test_opening),
        "strategy_opening_all": np.asarray(strategy_opening_all),
        "y_true_main": np.asarray(y_true_main),
        "y_pred_main": np.asarray(y_pred_main),
        "mass_main": np.asarray(mass_main),
        "opening_main": np.asarray(opening_main),
        "strategy_opening_main": np.asarray(strategy_opening_main),
        "policy_mask": np.asarray(policy_mask),
        "opening_dist_all": opening_dist_all,
        "opening_dist_main": opening_dist_main,
        "artifact_model_path": model_path,
        "artifact_meta_path": meta_path,
    }


# =========================
# compare / save
# =========================
def _validate_same_length(arr1, arr2, name1, name2):
    if len(arr1) != len(arr2):
        raise ValueError(
            f"{name1} 与 {name2} 长度不一致：{name1}={len(arr1)}, {name2}={len(arr2)}"
        )


def _validate_same_values(arr1, arr2, name1, name2, atol=1e-8, rtol=1e-6):
    a1 = np.asarray(arr1).reshape(-1)
    a2 = np.asarray(arr2).reshape(-1)
    _validate_same_length(a1, a2, name1, name2)
    if not np.allclose(a1, a2, atol=atol, rtol=rtol, equal_nan=False):
        bad = np.where(~np.isclose(a1, a2, atol=atol, rtol=rtol, equal_nan=False))[0]
        idx = int(bad[0])
        raise ValueError(
            f"{name1} 与 {name2} 数值不一致，首个差异 index={idx}："
            f"{name1}={a1[idx]}, {name2}={a2[idx]}"
        )


def _validate_same_mask(mask1, mask2, name1, name2):
    m1 = np.asarray(mask1).astype(bool).reshape(-1)
    m2 = np.asarray(mask2).astype(bool).reshape(-1)
    _validate_same_length(m1, m2, name1, name2)
    if not np.array_equal(m1, m2):
        bad = np.where(m1 != m2)[0]
        idx = int(bad[0])
        raise ValueError(
            f"{name1} 与 {name2} 不一致，首个差异 index={idx}：{name1}={m1[idx]}, {name2}={m2[idx]}"
        )


def _fmt_float(x, ndigits=6):
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NaN"
        return f"{float(x):.{ndigits}g}"
    return str(x)


def run_inverse_opening_holdout_compare(output_dir, data_path="data/dataset.xlsx", seed=42):
    print("\n" + "=" * 72)
    print("开始反向模型对比：opening-holdout (test = all 20/35/50 mm)")
    print("=" * 72)

    mlp_res = train_and_eval_inverse_mlp_holdout(
        data_path=data_path,
        random_state=seed,
        save_outputs_dir=None,
    )
    grnn_res = train_and_eval_inverse_grnn_holdout(
        data_path=data_path,
        random_state=seed,
        save_outputs_dir=None,
    )
    kan_res = train_and_eval_inverse_kan_holdout(
        data_path=data_path,
        seed=seed,
        save_artifacts=False,
        save_outputs_dir=None,
    )

    y_true_all_mlp = _to_1d_array(mlp_res["y_true_all"], "inverse_MLP y_true_all")
    y_true_all_grnn = _to_1d_array(grnn_res["y_true_all"], "inverse_GRNN y_true_all")
    y_true_all_kan = _to_1d_array(kan_res["y_true_all"], "inverse_KAN y_true_all")

    y_pred_all_mlp = _to_1d_array(mlp_res["y_pred_all"], "inverse_MLP y_pred_all")
    y_pred_all_grnn = _to_1d_array(grnn_res["y_pred_all"], "inverse_GRNN y_pred_all")
    y_pred_all_kan = _to_1d_array(kan_res["y_pred_all"], "inverse_KAN y_pred_all")

    opening_all_mlp = _to_1d_array(mlp_res["opening_all"], "inverse_MLP opening_all")
    opening_all_grnn = _to_1d_array(grnn_res["opening_all"], "inverse_GRNN opening_all")
    opening_all_kan = _to_1d_array(kan_res["opening_all"], "inverse_KAN opening_all")

    mass_all_mlp = _to_1d_array(mlp_res["mass_all"], "inverse_MLP mass_all")
    mass_all_grnn = _to_1d_array(grnn_res["mass_all"], "inverse_GRNN mass_all")
    mass_all_kan = _to_1d_array(kan_res["mass_all"], "inverse_KAN mass_all")

    strat_open_all_mlp = _to_1d_array(mlp_res["strategy_opening_all"], "inverse_MLP strategy_opening_all")
    strat_open_all_grnn = _to_1d_array(grnn_res["strategy_opening_all"], "inverse_GRNN strategy_opening_all")
    strat_open_all_kan = _to_1d_array(kan_res["strategy_opening_all"], "inverse_KAN strategy_opening_all")

    policy_mask_mlp = np.asarray(mlp_res["policy_mask"]).astype(bool).reshape(-1)
    policy_mask_grnn = np.asarray(grnn_res["policy_mask"]).astype(bool).reshape(-1)
    policy_mask_kan = np.asarray(kan_res["policy_mask"]).astype(bool).reshape(-1)

    _validate_same_values(y_true_all_mlp, y_true_all_grnn, "inverse_MLP y_true_all", "inverse_GRNN y_true_all")
    _validate_same_values(y_true_all_mlp, y_true_all_kan, "inverse_MLP y_true_all", "inverse_KAN y_true_all")

    _validate_same_values(mass_all_mlp, mass_all_grnn, "inverse_MLP mass_all", "inverse_GRNN mass_all")
    _validate_same_values(mass_all_mlp, mass_all_kan, "inverse_MLP mass_all", "inverse_KAN mass_all")

    _validate_same_values(opening_all_mlp, opening_all_grnn, "inverse_MLP opening_all", "inverse_GRNN opening_all")
    _validate_same_values(opening_all_mlp, opening_all_kan, "inverse_MLP opening_all", "inverse_KAN opening_all")

    _validate_same_values(
        strat_open_all_mlp,
        strat_open_all_grnn,
        "inverse_MLP strategy_opening_all",
        "inverse_GRNN strategy_opening_all",
    )
    _validate_same_values(
        strat_open_all_mlp,
        strat_open_all_kan,
        "inverse_MLP strategy_opening_all",
        "inverse_KAN strategy_opening_all",
    )

    _validate_same_mask(policy_mask_mlp, policy_mask_grnn, "inverse_MLP policy_mask", "inverse_GRNN policy_mask")
    _validate_same_mask(policy_mask_mlp, policy_mask_kan, "inverse_MLP policy_mask", "inverse_KAN policy_mask")

    metrics = [
        {
            "Task": "inverse_opening_holdout",
            "Model": "inverse_MLP",
            "DisplayName": "inverse_MLP",
            "ArchitectureNote": "test = all 20/35/50 mm openings",
            "R2_main": mlp_res["r2_main"],
            "ARE_main(%)": mlp_res["are_main"],
            "n_main": mlp_res["n_main"],
            "main_ratio": mlp_res["main_ratio"],
            "R2_all": mlp_res["r2_all"],
            "ARE_all(%)": mlp_res["are_all"],
            "n_all": mlp_res["n_all"],
            "Hyperparams": (
                f"hidden={mlp_res.get('best_hidden')}, "
                f"alpha={_fmt_float(mlp_res.get('best_alpha'))}"
            ),
        },
        {
            "Task": "inverse_opening_holdout",
            "Model": "inverse_GRNN",
            "DisplayName": "inverse_GRNN",
            "ArchitectureNote": "test = all 20/35/50 mm openings",
            "R2_main": grnn_res["r2_main"],
            "ARE_main(%)": grnn_res["are_main"],
            "n_main": grnn_res["n_main"],
            "main_ratio": grnn_res["main_ratio"],
            "R2_all": grnn_res["r2_all"],
            "ARE_all(%)": grnn_res["are_all"],
            "n_all": grnn_res["n_all"],
            "Hyperparams": f"sigma={_fmt_float(grnn_res.get('best_sigma'))}",
        },
        {
            "Task": "inverse_opening_holdout",
            "Model": "inverse_KAN",
            "DisplayName": "inverse_KAN",
            "ArchitectureNote": "repaired spline/grid KAN; test = all 20/35/50 mm openings",
            "R2_main": kan_res["r2_main"],
            "ARE_main(%)": kan_res["are_main"],
            "n_main": kan_res["n_main"],
            "main_ratio": kan_res["main_ratio"],
            "R2_all": kan_res["r2_all"],
            "ARE_all(%)": kan_res["are_all"],
            "n_all": kan_res["n_all"],
            "Hyperparams": (
                f"hidden={kan_res.get('best_hidden_dim')}, "
                f"lr={_fmt_float(kan_res.get('best_lr'))}, "
                f"wd={_fmt_float(kan_res.get('best_weight_decay'))}"
            ),
        },
    ]
    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "inverse_opening_holdout_metrics.csv")
    save_dataframe(df_metrics, metrics_path)

    df_all = pd.DataFrame({
        "target_mass_g_min": mass_all_mlp,
        "actual_opening_mm": opening_all_mlp,
        "strategy_opening_mm": strat_open_all_mlp,
        "true_speed_r_min": y_true_all_mlp,
        "inverse_MLP_pred": y_pred_all_mlp,
        "inverse_GRNN_pred": y_pred_all_grnn,
        "inverse_KAN_pred": y_pred_all_kan,
        "policy_match": policy_mask_mlp.astype(int),
    })
    all_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_all.csv")
    save_dataframe(df_all, all_path)

    main_idx = np.where(policy_mask_mlp)[0]
    df_main = pd.DataFrame({
        "target_mass_g_min": mass_all_mlp[main_idx],
        "actual_opening_mm": opening_all_mlp[main_idx],
        "strategy_opening_mm": strat_open_all_mlp[main_idx],
        "true_speed_r_min": y_true_all_mlp[main_idx],
        "inverse_MLP_pred": y_pred_all_mlp[main_idx],
        "inverse_GRNN_pred": y_pred_all_grnn[main_idx],
        "inverse_KAN_pred": y_pred_all_kan[main_idx],
    })
    main_path = os.path.join(output_dir, "inverse_opening_holdout_predictions_main.csv")
    save_dataframe(df_main, main_path)

    print(f"反向指标已保存：{metrics_path}")
    print(f"反向全测试集预测已保存：{all_path}")
    print(f"反向主结果子集预测已保存：{main_path}")

    return {
        "metrics_path": metrics_path,
        "all_path": all_path,
        "main_path": main_path,
    }


def main():
    data_path = "data/dataset.xlsx"
    seed = 42
    run_dir = create_run_dir("evaluate_inverse_opening_holdout")

    manifest_path = write_manifest(
        run_dir,
        script_name="evaluate_inverse_opening_holdout.py",
        data_path=data_path,
        seed=seed,
        params={
            "test_openings_mm": [20.0, 35.0, 50.0],
            "threshold_low_mid": THRESHOLD_LOW_MID,
            "threshold_mid_high": THRESHOLD_MID_HIGH,
            "note": "all 20/35/50 mm samples are excluded from training and used only as inverse-task test set",
        },
    )

    print(f"\n本次运行目录：{run_dir}")
    print(f"Manifest：{manifest_path}")

    outputs = run_inverse_opening_holdout_compare(run_dir, data_path=data_path, seed=seed)

    append_manifest_outputs(
        run_dir,
        [
            {"path": os.path.relpath(outputs["metrics_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(outputs["all_path"], run_dir).replace("\\", "/")},
            {"path": os.path.relpath(outputs["main_path"], run_dir).replace("\\", "/")},
        ],
    )

    print("\n全部结果已统一输出到：")
    print(run_dir)


if __name__ == "__main__":
    main()