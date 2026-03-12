# -*- coding: utf-8 -*-
"""
train_kan.py

正向 KAN：
- 默认不再写仓库根目录 results_kan.csv
- 默认不再写顶层 path/
- 独立运行时输出到 runs/<timestamp>_train_kan/
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

from common_utils import load_data, get_train_val_test_indices, average_relative_error
from run_utils import append_manifest_outputs, create_run_dir, ensure_dir, save_dataframe, write_manifest


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


class FertilizerKAN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=1):
        super().__init__()
        self.kan1 = KANLayer(input_dim, hidden_dim, grid_size=10)
        self.kan2 = KANLayer(hidden_dim, output_dim, grid_size=10)

    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        return x


def save_forward_artifacts(
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
    search_epochs,
    gamma,
    best_val_r2,
    x_min,
    x_max,
    y_min,
    y_max,
    x_train_raw,
    y_train_raw,
    idx_tr,
    idx_val,
    idx_te,
):
    torch.save(model.state_dict(), model_path)

    meta = {
        "artifact_type": "forward_model_bundle",
        "model_name": "KAN",
        "model_class": "FertilizerKAN",
        "weight_path": model_path.replace("\\", "/"),
        "data_path": data_path,
        "seed": int(seed),
        "hyperparameters": {
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "search_epochs": int(search_epochs),
            "gamma": float(gamma),
        },
        "validation_result": {
            "best_val_r2": float(best_val_r2),
        },
        "normalization_params": {
            "X_min": np.asarray(x_min).tolist(),
            "X_max": np.asarray(x_max).tolist(),
            "y_min": float(np.asarray(y_min).reshape(-1)[0]),
            "y_max": float(np.asarray(y_max).reshape(-1)[0]),
        },
        "training_domain": {
            "opening_min": float(x_train_raw[:, 0].min()),
            "opening_max": float(x_train_raw[:, 0].max()),
            "speed_min": float(x_train_raw[:, 1].min()),
            "speed_max": float(x_train_raw[:, 1].max()),
            "mass_min": float(y_train_raw.min()),
            "mass_max": float(y_train_raw.max()),
        },
        "split_info": {
            "train_size": int(len(idx_tr)),
            "val_size": int(len(idx_val)),
            "test_size": int(len(idx_te)),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def train_and_eval_kan(
    data_path="data/dataset.xlsx",
    hidden_dim_candidates=None,
    lr_candidates=None,
    weight_decay_candidates=None,
    epochs=600,
    search_epochs=300,
    gamma=0.99,
    save_csv_path=None,
    seed=42,
    save_artifacts=False,
    artifact_dir=None,
    weight_filename="kan_forward.pth",
    meta_filename="kan_forward_meta.json",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== 训练 KAN ===")
    print("Using:", device)
    print("Random seed:", seed)

    if hidden_dim_candidates is None:
        hidden_dim_candidates = [4, 8, 16]
    if lr_candidates is None:
        lr_candidates = [0.01, 0.005]
    if weight_decay_candidates is None:
        weight_decay_candidates = [1e-4, 1e-5]

    X, y = load_data(data_path)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X, y=y)

    X_train_raw, y_train_raw = X[idx_tr], y[idx_tr]
    X_val_raw, y_val_raw = X[idx_val], y[idx_val]
    X_test_raw, y_test_raw = X[idx_te], y[idx_te]

    X_min = X_train_raw.min(axis=0, keepdims=True)
    X_max = X_train_raw.max(axis=0, keepdims=True)

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + EPS)

    X_train_np = norm_x(X_train_raw)
    X_val_np = norm_x(X_val_raw)
    X_test_np = norm_x(X_test_raw)

    y_min = y_train_raw.min(keepdims=True)
    y_max = y_train_raw.max(keepdims=True)

    y_train_np = (y_train_raw - y_min) / (y_max - y_min + EPS)
    y_val_np = (y_val_raw - y_min) / (y_max - y_min + EPS)

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(device)
    y_val_t = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1).to(device)

    criterion = nn.MSELoss()
    best_r2_val = -np.inf
    best_cfg = None

    for hidden_dim in hidden_dim_candidates:
        for lr in lr_candidates:
            for wd in weight_decay_candidates:
                set_seed(seed)

                model = FertilizerKAN(
                    input_dim=2,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                ).to(device)

                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

                for _ in range(search_epochs):
                    model.train()
                    optimizer.zero_grad()
                    pred_train_norm = model(X_train_t)
                    loss = criterion(pred_train_norm, y_train_t)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                model.eval()
                with torch.no_grad():
                    pred_val_norm = model(X_val_t).cpu().numpy().reshape(-1)

                y_pred_val = pred_val_norm * (y_max - y_min) + y_min
                r2_val = r2_score(y_val_raw, y_pred_val)

                if r2_val > best_r2_val:
                    best_r2_val = r2_val
                    best_cfg = (hidden_dim, lr, wd)

    if best_cfg is None:
        raise RuntimeError("KAN 超参数搜索失败，未找到有效配置。")

    hidden_dim_best, lr_best, wd_best = best_cfg

    print(
        f"KAN 最优超参数：hidden_dim={hidden_dim_best}, "
        f"lr={lr_best}, weight_decay={wd_best}, val R²={best_r2_val:.6f}"
    )

    X_train_val_t = torch.cat([X_train_t, X_val_t], dim=0)
    y_train_val_t = torch.cat([y_train_t, y_val_t], dim=0)

    set_seed(seed)
    model_final = FertilizerKAN(
        input_dim=2,
        hidden_dim=hidden_dim_best,
        output_dim=1,
    ).to(device)

    optimizer = optim.AdamW(model_final.parameters(), lr=lr_best, weight_decay=wd_best)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    print("开始训练 KAN 最终模型 ...")
    for epoch in range(epochs):
        model_final.train()
        optimizer.zero_grad()
        pred_train_norm = model_final(X_train_val_t)
        loss = criterion(pred_train_norm, y_train_val_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss={loss.item():.6f}")

    model_final.eval()
    with torch.no_grad():
        pred_test_norm = model_final(X_test_t).cpu().numpy().reshape(-1)

    y_pred_kan = pred_test_norm * (y_max - y_min) + y_min
    kan_r2 = r2_score(y_test_raw, y_pred_kan)
    kan_are = average_relative_error(y_test_raw, y_pred_kan)

    print("\n===== KAN 结果 =====")
    print(f"R²  = {kan_r2:.6f}")
    print(f"ARE = {kan_are:.6f} %")

    if save_csv_path is not None:
        df_out = pd.DataFrame({
            "true": y_test_raw,
            "KAN_pred": y_pred_kan,
        })
        save_dataframe(df_out, save_csv_path)
        print(f"预测文件已保存：{save_csv_path}")

    model_path = None
    meta_path = None
    if save_artifacts:
        if artifact_dir is None:
            raise ValueError("save_artifacts=True 时必须提供 artifact_dir")
        ensure_dir(artifact_dir)
        model_path = os.path.join(artifact_dir, weight_filename)
        meta_path = os.path.join(artifact_dir, meta_filename)

        save_forward_artifacts(
            model_final,
            model_path=model_path,
            meta_path=meta_path,
            seed=seed,
            data_path=data_path,
            hidden_dim=hidden_dim_best,
            lr=lr_best,
            weight_decay=wd_best,
            epochs=epochs,
            search_epochs=search_epochs,
            gamma=gamma,
            best_val_r2=best_r2_val,
            x_min=X_min,
            x_max=X_max,
            y_min=y_min,
            y_max=y_max,
            x_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            idx_tr=idx_tr,
            idx_val=idx_val,
            idx_te=idx_te,
        )
        print(f"KAN 工件已保存：{artifact_dir}")

    return {
        "r2": kan_r2,
        "are": kan_are,
        "best_hidden_dim": hidden_dim_best,
        "best_lr": lr_best,
        "best_weight_decay": wd_best,
        "y_true": y_test_raw,
        "y_pred": y_pred_kan,
        "seed": seed,
        "artifact_model_path": model_path,
        "artifact_meta_path": meta_path,
    }


def main():
    run_dir = create_run_dir("train_kan")
    output_csv = os.path.join(run_dir, "results_kan.csv")
    artifact_dir = os.path.join(run_dir, "artifacts")

    write_manifest(
        run_dir,
        script_name="train_kan.py",
        data_path="data/dataset.xlsx",
        seed=42,
        params={
            "hidden_dim_candidates": [4, 8, 16],
            "lr_candidates": [0.01, 0.005],
            "weight_decay_candidates": [1e-4, 1e-5],
            "epochs": 600,
            "search_epochs": 300,
            "gamma": 0.99,
        },
    )

    res = train_and_eval_kan(
        save_csv_path=output_csv,
        seed=42,
        save_artifacts=True,
        artifact_dir=artifact_dir,
    )

    outputs = [{"path": "results_kan.csv"}]
    if res["artifact_model_path"] is not None:
        outputs.append({"path": "artifacts/kan_forward.pth"})
    if res["artifact_meta_path"] is not None:
        outputs.append({"path": "artifacts/kan_forward_meta.json"})

    append_manifest_outputs(run_dir, outputs)
    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()