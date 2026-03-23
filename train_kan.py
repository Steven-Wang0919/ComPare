# -*- coding: utf-8 -*-
"""
train_kan.py

Forward KAN with a shared fair tuning protocol and replayable artifact bundles.
"""

import gc
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

from common_utils import (
    average_relative_error,
    build_sample_tracking_columns,
    combine_train_val_indices,
    get_train_val_test_indices,
    load_data_with_metadata,
    validate_predefined_split_indices,
)
from fair_tuning import (
    ensure_fair_tuning_config,
    infer_inner_val_ratio,
    prepare_inner_cv,
    run_fair_tuning,
    tuning_config_to_dict,
)
from run_utils import (
    append_manifest_outputs,
    build_artifact_metadata,
    build_single_split_artifact_payload,
    build_split_indices_payload,
    build_tuning_protocol_payload,
    create_run_dir,
    ensure_dir,
    save_dataframe,
    save_test_slice,
    update_manifest_split_artifact,
    write_json,
    write_manifest,
)


EPS = 1e-8
DEFAULT_HIDDEN_DIM_CANDIDATES = [4, 8, 16, 32]
DEFAULT_LR_CANDIDATES = [1e-2, 5e-3, 1e-3]
DEFAULT_WEIGHT_DECAY_CANDIDATES = [1e-4, 1e-5]
MODEL_FILENAME = "model.pth"
META_FILENAME = "meta.json"


def _artifact_source_files():
    base_dir = os.path.dirname(__file__)
    return [
        __file__,
        os.path.join(base_dir, "common_utils.py"),
        os.path.join(base_dir, "fair_tuning.py"),
        os.path.join(base_dir, "run_utils.py"),
    ]


def _cleanup_torch_runtime(*, model=None, tensors=None, device=None):
    tensors = list(tensors or [])
    for tensor in tensors:
        try:
            del tensor
        except Exception:
            pass
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    gc.collect()
    device_name = str(device or "")
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


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


def _build_candidate_configs(
    hidden_dim_candidates=None,
    lr_candidates=None,
    weight_decay_candidates=None,
):
    hidden_dim_candidates = hidden_dim_candidates or DEFAULT_HIDDEN_DIM_CANDIDATES
    lr_candidates = lr_candidates or DEFAULT_LR_CANDIDATES
    weight_decay_candidates = weight_decay_candidates or DEFAULT_WEIGHT_DECAY_CANDIDATES
    return [
        {
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }
        for hidden_dim in hidden_dim_candidates
        for lr in lr_candidates
        for weight_decay in weight_decay_candidates
    ]


def _prepare_forward_arrays(X_train_raw, y_train_raw, X_eval_raw):
    X_min = X_train_raw.min(axis=0, keepdims=True)
    X_max = X_train_raw.max(axis=0, keepdims=True)
    y_min = y_train_raw.min(keepdims=True)
    y_max = y_train_raw.max(keepdims=True)

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + EPS)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min) + y_min

    arrays = {
        "X_train": norm_x(X_train_raw),
        "X_eval": norm_x(X_eval_raw),
        "y_train": (y_train_raw - y_min) / (y_max - y_min + EPS),
        "X_min": X_min,
        "X_max": X_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    arrays["denorm_y"] = denorm_y
    return arrays


def _train_forward_kan_model(
    X_train_np,
    y_train_np,
    *,
    hidden_dim,
    lr,
    weight_decay,
    epochs,
    gamma,
    device,
    seed,
):
    set_seed(seed)
    model = FertilizerKAN(input_dim=2, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred_train_norm = model(X_train_t)
        loss = criterion(pred_train_norm, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model


def _fit_predict_forward_kan(
    X_train_raw,
    y_train_raw,
    X_eval_raw,
    *,
    hidden_dim,
    lr,
    weight_decay,
    epochs,
    gamma,
    device,
    seed,
    return_model=False,
):
    arrays = _prepare_forward_arrays(X_train_raw, y_train_raw, X_eval_raw)
    model = _train_forward_kan_model(
        arrays["X_train"],
        arrays["y_train"],
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        gamma=gamma,
        device=device,
        seed=seed,
    )
    model.eval()
    X_eval_t = None
    with torch.no_grad():
        X_eval_t = torch.tensor(arrays["X_eval"], dtype=torch.float32).to(device)
        pred_eval_norm = model(X_eval_t).cpu().numpy().reshape(-1)

    y_pred_eval = arrays["denorm_y"](pred_eval_norm)
    if not return_model:
        _cleanup_torch_runtime(model=model, tensors=[X_eval_t], device=device)
        model = None
    return {
        "model": model,
        "y_pred_eval": y_pred_eval,
        "norm_stats": {
            "X_min": arrays["X_min"],
            "X_max": arrays["X_max"],
            "y_min": arrays["y_min"],
            "y_max": arrays["y_max"],
        },
    }


def save_forward_artifacts(
    model,
    model_path,
    meta_path,
    *,
    data_path,
    best_config,
    norm_stats,
    x_train_raw,
    y_train_raw,
    idx_tr,
    idx_val,
    idx_te,
    tuning_config,
    tuning_result,
    resolved_inner_splits=None,
    artifact_extra=None,
    artifact_source_files=None,
    x_test_raw=None,
    y_test_raw=None,
    save_test_slice_flag=False,
):
    torch.save(model.state_dict(), model_path)
    test_inputs_path = None
    test_targets_path = None
    if save_test_slice_flag and x_test_raw is not None and y_test_raw is not None:
        test_inputs_path, test_targets_path = save_test_slice(os.path.dirname(model_path), x_test_raw, y_test_raw)

    tuning_payload = build_tuning_protocol_payload(
        tuning_config_to_dict(tuning_config),
        inner_split_strategy=tuning_result["inner_split_strategy"],
        inner_split_meta=tuning_result["inner_split_meta"],
        inner_splits=resolved_inner_splits,
        tuning_seed=int(tuning_config.seed),
        n_repeats=int(tuning_result["inner_fold_count"]),
        inner_val_ratio=tuning_config.inner_val_ratio,
    )
    meta = build_artifact_metadata(
        artifact_type="model_bundle",
        task_name="forward",
        model_name="KAN",
        model_class="train_kan.FertilizerKAN",
        data_path=data_path,
        best_config=best_config,
        normalization_params=norm_stats,
        split_indices=build_split_indices_payload(idx_tr, idx_val, idx_te),
        tuning_protocol=tuning_payload,
        training_domain={
            "feature_names": ["opening_mm", "speed_r_min"],
            "target_name": "mass_g_min",
            "opening_min": float(x_train_raw[:, 0].min()),
            "opening_max": float(x_train_raw[:, 0].max()),
            "speed_min": float(x_train_raw[:, 1].min()),
            "speed_max": float(x_train_raw[:, 1].max()),
            "mass_min": float(y_train_raw.min()),
            "mass_max": float(y_train_raw.max()),
        },
        extra={
            "model_file": os.path.basename(model_path),
            "meta_file": os.path.basename(meta_path),
            "test_inputs_file": os.path.basename(test_inputs_path) if test_inputs_path is not None else None,
            "test_targets_file": os.path.basename(test_targets_path) if test_targets_path is not None else None,
            **dict(artifact_extra or {}),
        },
        source_files=artifact_source_files or _artifact_source_files(),
    )
    write_json(meta_path, meta)
    return test_inputs_path, test_targets_path


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
    weight_filename=MODEL_FILENAME,
    meta_filename=META_FILENAME,
    split_indices=None,
    tuning_config=None,
    save_tuning_records_path=None,
    inner_splits=None,
    inner_split_strategy=None,
    inner_split_meta=None,
    save_test_slice=False,
    artifact_extra=None,
    artifact_source_files=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== 训练 KAN（公平调参协议） ===")
    print("Using:", device)
    print("Random seed:", seed)

    X, y, sample_meta = load_data_with_metadata(data_path)
    if split_indices is None:
        idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X, y=y, random_state=seed)
    else:
        idx_tr, idx_val, idx_te = validate_predefined_split_indices(
            len(X), split_indices[0], split_indices[1], split_indices[2]
        )

    inner_val_ratio = None if inner_splits is not None else infer_inner_val_ratio(idx_tr, idx_val)
    tuning_config = ensure_fair_tuning_config(
        tuning_config,
        seed=seed,
        inner_val_ratio=inner_val_ratio,
    )

    dev_idx = combine_train_val_indices(idx_tr, idx_val)
    X_dev_raw = X[dev_idx]
    y_dev_raw = y[dev_idx]
    X_test_raw = X[idx_te]
    y_test_raw = y[idx_te]
    test_tracking = build_sample_tracking_columns(sample_meta, idx_te)
    split_indices_payload = build_single_split_artifact_payload(
        idx_tr,
        idx_val,
        idx_te,
        n_samples=len(X),
    )

    inner_splits, inner_split_strategy, inner_split_meta = prepare_inner_cv(
        X_dev_raw,
        y_dev_raw,
        tuning_config,
        inner_splits=inner_splits,
        inner_split_strategy=inner_split_strategy,
        inner_split_meta=inner_split_meta,
    )
    candidate_configs = _build_candidate_configs(
        hidden_dim_candidates,
        lr_candidates,
        weight_decay_candidates,
    )

    def eval_candidate_fn(*, config, idx_train, idx_val, fold_id, split_meta):
        del split_meta
        res = _fit_predict_forward_kan(
            X_dev_raw[idx_train],
            y_dev_raw[idx_train],
            X_dev_raw[idx_val],
            hidden_dim=config["hidden_dim"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            epochs=search_epochs,
            gamma=gamma,
            device=device,
            seed=int(tuning_config.seed) + int(fold_id),
            return_model=False,
        )
        y_pred_val = res["y_pred_eval"]
        y_true_val = y_dev_raw[idx_val]
        return {
            "val_r2": float(r2_score(y_true_val, y_pred_val)),
            "val_are": float(average_relative_error(y_true_val, y_pred_val)),
        }

    tuning_result = run_fair_tuning(
        candidate_configs=candidate_configs,
        inner_splits=inner_splits,
        eval_candidate_fn=eval_candidate_fn,
        tuning_config=tuning_config,
        model_name="KAN",
        task_name="forward",
        inner_split_strategy=inner_split_strategy,
        inner_split_meta=inner_split_meta,
    )
    best_config = tuning_result["best_config"]
    best_summary = tuning_result["candidate_summaries"][tuning_result["best_candidate_idx"]]

    final_fit = _fit_predict_forward_kan(
        X_dev_raw,
        y_dev_raw,
        X_test_raw,
        hidden_dim=best_config["hidden_dim"],
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
        epochs=epochs,
        gamma=gamma,
        device=device,
        seed=int(tuning_config.seed),
        return_model=save_artifacts,
    )
    model_final = final_fit["model"]
    y_pred_kan = final_fit["y_pred_eval"]

    kan_r2 = float(r2_score(y_test_raw, y_pred_kan))
    kan_are = float(average_relative_error(y_test_raw, y_pred_kan))

    print(
        "KAN 最优超参数："
        f"hidden_dim={best_config['hidden_dim']}, "
        f"lr={best_config['lr']}, "
        f"weight_decay={best_config['weight_decay']}, "
        f"mean val R2={best_summary['mean_val_r2']:.6f}"
    )
    print("\n===== KAN 结果 =====")
    print(f"R2  = {kan_r2:.6f}")
    print(f"ARE = {kan_are:.6f} %")

    if save_csv_path is not None:
        df_out = pd.DataFrame({
            **test_tracking,
            "true": y_test_raw,
            "KAN_pred": y_pred_kan,
        })
        save_dataframe(df_out, save_csv_path)
        print(f"预测文件已保存：{save_csv_path}")

    if save_tuning_records_path is not None:
        save_dataframe(pd.DataFrame(tuning_result["tuning_records"]), save_tuning_records_path)
        print(f"调参审计文件已保存：{save_tuning_records_path}")

    model_path = None
    meta_path = None
    artifact_test_inputs_path = None
    artifact_test_targets_path = None
    if save_artifacts:
        if artifact_dir is None:
            raise ValueError("save_artifacts=True 时必须提供 artifact_dir")
        ensure_dir(artifact_dir)
        model_path = os.path.join(artifact_dir, weight_filename)
        meta_path = os.path.join(artifact_dir, meta_filename)
        artifact_test_inputs_path, artifact_test_targets_path = save_forward_artifacts(
            model_final,
            model_path=model_path,
            meta_path=meta_path,
            data_path=data_path,
            best_config={
                "hidden_dim": int(best_config["hidden_dim"]),
                "lr": float(best_config["lr"]),
                "weight_decay": float(best_config["weight_decay"]),
                "epochs": int(epochs),
                "search_epochs": int(search_epochs),
                "gamma": float(gamma),
                "best_val_r2": float(best_summary["mean_val_r2"]),
            },
            norm_stats=final_fit["norm_stats"],
            x_train_raw=X_dev_raw,
            y_train_raw=y_dev_raw,
            idx_tr=idx_tr,
            idx_val=idx_val,
            idx_te=idx_te,
            tuning_config=tuning_config,
            tuning_result=tuning_result,
            resolved_inner_splits=(
                inner_splits if tuning_result["inner_split_strategy"] != "repeated_random" else None
            ),
            artifact_extra=artifact_extra,
            artifact_source_files=artifact_source_files,
            x_test_raw=X_test_raw,
            y_test_raw=y_test_raw,
            save_test_slice_flag=save_test_slice,
        )
        print(f"KAN 工件已保存：{artifact_dir}")

    _cleanup_torch_runtime(model=model_final, device=device)

    return {
        "r2": kan_r2,
        "are": kan_are,
        "best_hidden_dim": int(best_config["hidden_dim"]),
        "best_lr": float(best_config["lr"]),
        "best_weight_decay": float(best_config["weight_decay"]),
        "best_config": {
            "hidden_dim": int(best_config["hidden_dim"]),
            "lr": float(best_config["lr"]),
            "weight_decay": float(best_config["weight_decay"]),
        },
        "test_sample_id": test_tracking["sample_id"],
        "test_source_row_number": test_tracking["source_row_number"],
        "y_true": y_test_raw,
        "y_pred": y_pred_kan,
        "x_test_raw": X_test_raw,
        "seed": seed,
        "split_indices_payload": split_indices_payload,
        "artifact_dir": artifact_dir,
        "artifact_model_path": model_path,
        "artifact_meta_path": meta_path,
        "artifact_test_inputs_path": artifact_test_inputs_path,
        "artifact_test_targets_path": artifact_test_targets_path,
        "tuning_protocol": tuning_config_to_dict(tuning_config),
        "trial_budget": int(tuning_config.n_candidates),
        "validation_repeats": int(tuning_result["inner_fold_count"]),
        "selection_metric": tuning_config.selection_metric,
        "tie_break_metric": tuning_config.tie_break_metric,
        "inner_split_strategy": tuning_result["inner_split_strategy"],
        "inner_split_meta": tuning_result["inner_split_meta"],
        "inner_fold_count": int(tuning_result["inner_fold_count"]),
        "tuning_records": tuning_result["tuning_records"],
        "candidate_summaries": tuning_result["candidate_summaries"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the forward KAN baseline.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = int(args.seed)
    run_dir = create_run_dir("train_kan")
    output_csv = os.path.join(run_dir, "results_kan.csv")
    tuning_csv = os.path.join(run_dir, "tuning_records_kan.csv")
    artifact_dir = os.path.join(run_dir, "artifacts", "forward", "KAN")
    data_path = "data/dataset.xlsx"
    X, y, _ = load_data_with_metadata(data_path)
    split_indices = get_train_val_test_indices(X=X, y=y, random_state=seed)
    split_payload = build_single_split_artifact_payload(
        split_indices[0],
        split_indices[1],
        split_indices[2],
        n_samples=len(X),
    )

    write_manifest(
        run_dir,
        script_name="train_kan.py",
        data_path=data_path,
        seed=seed,
        params={
            "hidden_dim_candidates": DEFAULT_HIDDEN_DIM_CANDIDATES,
            "lr_candidates": DEFAULT_LR_CANDIDATES,
            "weight_decay_candidates": DEFAULT_WEIGHT_DECAY_CANDIDATES,
            "epochs": 600,
            "search_epochs": 300,
            "gamma": 0.99,
            "fair_tuning": {
                "n_candidates": 24,
                "n_repeats": 5,
                "selection_metric": "mean_val_r2",
                "tie_break_metric": "mean_val_are",
                "budget_profile": "high",
            },
        },
        source_files=_artifact_source_files(),
        split_payload=split_payload,
    )

    res = train_and_eval_kan(
        data_path=data_path,
        save_csv_path=output_csv,
        seed=seed,
        split_indices=split_indices,
        save_artifacts=True,
        artifact_dir=artifact_dir,
        save_tuning_records_path=tuning_csv,
        save_test_slice=True,
        artifact_extra={
            "run_dir": run_dir.replace("\\", "/"),
            "reference_output": {
                "path": "results_kan.csv",
                "prediction_column": "KAN_pred",
                "target_column": "true",
            },
        },
    )

    update_manifest_split_artifact(run_dir, split_payload=res["split_indices_payload"])

    outputs = [
        {"path": "results_kan.csv"},
        {"path": "tuning_records_kan.csv"},
    ]
    if res["artifact_model_path"] is not None:
        outputs.append({"path": "artifacts/forward/KAN/model.pth"})
    if res["artifact_meta_path"] is not None:
        outputs.append({"path": "artifacts/forward/KAN/meta.json"})
    if res["artifact_test_inputs_path"] is not None:
        outputs.append({"path": "artifacts/forward/KAN/test_inputs.npy"})
    if res["artifact_test_targets_path"] is not None:
        outputs.append({"path": "artifacts/forward/KAN/test_targets.npy"})

    append_manifest_outputs(run_dir, outputs)
    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()
