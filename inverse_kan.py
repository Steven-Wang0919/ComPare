# -*- coding: utf-8 -*-
"""
inverse_kan.py

反向 KAN（结构修复版）
- compare_all 调用时可 save_artifacts=False，避免污染仓库
- 独立运行时统一输出到 runs/<timestamp>_inverse_kan/
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, average_relative_error
from run_utils import append_manifest_outputs, create_run_dir, ensure_dir, save_dataframe, write_manifest


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0
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
    stats["other"] = int(len(openings) - sum(stats[k] for k in ["20mm", "35mm", "50mm"]))
    return stats


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

        arange_left = torch.arange(self.spline_order, 0, -1, device=device, dtype=grid.dtype).unsqueeze(0)
        left_pad = grid[:, 0:1] - arange_left * h

        arange_right = torch.arange(1, self.spline_order + 1, device=device, dtype=grid.dtype).unsqueeze(0)
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


def train_and_eval_inverse_kan_v2(
    data_path="data/dataset.xlsx",
    hidden_dim_candidates=None,
    lr_candidates=None,
    weight_decay_candidates=None,
    epochs=1000,
    device=None,
    seed=42,
    save_artifacts=False,
    artifact_dir=None,
    weight_filename="kan_inverse.pth",
    meta_filename="kan_inverse_meta.json",
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

    print("\n=== 训练 反向 KAN（结构修复 + train-only normalization） ===")
    print(f"使用设备: {device}")

    X_raw, y_raw = load_data(data_path)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X_raw, y=y_raw)

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]

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

                if val_r2 > best_val_r2:
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


def main():
    run_dir = create_run_dir("inverse_kan")
    artifact_dir = os.path.join(run_dir, "artifacts")

    write_manifest(
        run_dir,
        script_name="inverse_kan.py",
        data_path="data/dataset.xlsx",
        seed=42,
        params={
            "hidden_dim_candidates": [8, 16, 32],
            "lr_candidates": [1e-2, 5e-3, 1e-3],
            "weight_decay_candidates": [0.0, 1e-6, 1e-5, 1e-4],
            "epochs": 1000,
        },
    )

    res = train_and_eval_inverse_kan_v2(
        seed=42,
        save_outputs_dir=run_dir,
        save_artifacts=True,
        artifact_dir=artifact_dir,
    )

    outputs = [
        {"path": "inverse_kan_predictions_all.csv"},
        {"path": "inverse_kan_predictions_main.csv"},
    ]
    if res["artifact_model_path"] is not None:
        outputs.append({"path": "artifacts/kan_inverse.pth"})
    if res["artifact_meta_path"] is not None:
        outputs.append({"path": "artifacts/kan_inverse_meta.json"})

    append_manifest_outputs(run_dir, outputs)
    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()