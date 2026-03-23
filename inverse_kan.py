# -*- coding: utf-8 -*-
"""
inverse_kan.py

Inverse KAN with a shared fair tuning protocol and replayable artifact bundles.
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
    build_inner_repeated_splits,
    ensure_fair_tuning_config,
    infer_inner_val_ratio,
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
from policy_config import (
    POLICY_LABEL,
    POLICY_LOW_MID_THRESHOLD,
    POLICY_MID_HIGH_THRESHOLD,
    POLICY_TARGET_OPENINGS,
    select_policy_opening,
)


EPS = 1e-8
DEFAULT_HIDDEN_DIM_CANDIDATES = [8, 16, 32, 64]
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
        os.path.join(base_dir, "policy_config.py"),
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


def select_optimal_opening(target_mass: float) -> float:
    return select_policy_opening(target_mass)


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


def _count_openings(openings, opening_values=POLICY_TARGET_OPENINGS, atol=0.1):
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


def _prepare_inverse_arrays(X_train_raw, y_train_raw, X_eval_raw):
    x_min = X_train_raw.min(axis=0, keepdims=True)
    x_max = X_train_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_raw.min())
    y_max = float(y_train_raw.max())

    def norm_x(x):
        return (x - x_min) / (x_max - x_min + EPS)

    def denorm_y(y_norm):
        return y_norm * (y_max - y_min + EPS) + y_min

    arrays = {
        "X_train": norm_x(X_train_raw),
        "X_eval": norm_x(X_eval_raw),
        "y_train": (y_train_raw - y_min) / (y_max - y_min + EPS),
        "X_min": x_min,
        "X_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    arrays["denorm_y"] = denorm_y
    return arrays


def _train_inverse_kan_model(
    X_train,
    y_train,
    *,
    hidden_dim,
    lr,
    weight_decay,
    epochs,
    device,
    seed,
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


def _fit_predict_inverse_kan(
    X_train_raw,
    y_train_raw,
    X_eval_raw,
    *,
    hidden_dim,
    lr,
    weight_decay,
    epochs,
    device,
    seed,
    return_model=False,
):
    arrays = _prepare_inverse_arrays(X_train_raw, y_train_raw, X_eval_raw)
    model = _train_inverse_kan_model(
        arrays["X_train"],
        arrays["y_train"],
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        device=device,
        seed=seed,
    )

    model.eval()
    X_eval_t = None
    with torch.no_grad():
        X_eval_t = torch.tensor(arrays["X_eval"], dtype=torch.float32, device=device)
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


def save_inverse_artifacts(
    model,
    model_path,
    meta_path,
    *,
    data_path,
    best_config,
    norm_stats,
    x_train_domain_raw,
    y_train_domain_raw,
    idx_tr,
    idx_val,
    idx_te,
    tuning_config,
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
        inner_split_strategy="repeated_random",
        inner_split_meta={},
        tuning_seed=int(tuning_config.seed),
        n_repeats=int(tuning_config.n_repeats),
        inner_val_ratio=tuning_config.inner_val_ratio,
    )
    meta = build_artifact_metadata(
        artifact_type="model_bundle",
        task_name="inverse",
        model_name="inverse_KAN",
        model_class="inverse_kan.InverseKANModel",
        data_path=data_path,
        best_config=best_config,
        normalization_params=norm_stats,
        split_indices=build_split_indices_payload(idx_tr, idx_val, idx_te),
        tuning_protocol=tuning_payload,
        training_domain={
            "feature_names": ["target_mass_g_min", "opening_mm"],
            "target_name": "speed_r_min",
            "target_mass_min": float(x_train_domain_raw[:, 0].min()),
            "target_mass_max": float(x_train_domain_raw[:, 0].max()),
            "opening_min": float(x_train_domain_raw[:, 1].min()),
            "opening_max": float(x_train_domain_raw[:, 1].max()),
            "speed_min": float(y_train_domain_raw.min()),
            "speed_max": float(y_train_domain_raw.max()),
        },
        extra={
            "model_file": os.path.basename(model_path),
            "meta_file": os.path.basename(meta_path),
            "test_inputs_file": os.path.basename(test_inputs_path) if test_inputs_path is not None else None,
            "test_targets_file": os.path.basename(test_targets_path) if test_targets_path is not None else None,
            "policy": {
                "label": POLICY_LABEL,
                "target_openings_mm": list(POLICY_TARGET_OPENINGS),
                "threshold_low_mid": POLICY_LOW_MID_THRESHOLD,
                "threshold_mid_high": POLICY_MID_HIGH_THRESHOLD,
            },
            **dict(artifact_extra or {}),
        },
        source_files=artifact_source_files or _artifact_source_files(),
    )
    write_json(meta_path, meta)
    return test_inputs_path, test_targets_path


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
    weight_filename=MODEL_FILENAME,
    meta_filename=META_FILENAME,
    save_outputs_dir=None,
    split_indices=None,
    tuning_config=None,
    save_tuning_records_path=None,
    save_test_slice=False,
    artifact_extra=None,
    artifact_source_files=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== 训练 反向 KAN（公平调参协议） ===")
    print(f"使用设备: {device}")

    X_raw, y_raw, sample_meta = load_data_with_metadata(data_path)
    if split_indices is None:
        idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X_raw, y=y_raw, random_state=seed)
    else:
        idx_tr, idx_val, idx_te = validate_predefined_split_indices(
            len(X_raw), split_indices[0], split_indices[1], split_indices[2]
        )

    inner_val_ratio = infer_inner_val_ratio(idx_tr, idx_val)
    tuning_config = ensure_fair_tuning_config(
        tuning_config,
        seed=seed,
        inner_val_ratio=inner_val_ratio,
    )

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    dev_idx = combine_train_val_indices(idx_tr, idx_val)
    X_dev_raw = X_inv_all[dev_idx]
    y_dev_raw = y_inv_all[dev_idx]
    X_test_raw = X_inv_all[idx_te]
    y_test_speed_true = y_inv_all[idx_te]

    inner_splits = build_inner_repeated_splits(X_dev_raw, y_dev_raw, tuning_config)
    candidate_configs = _build_candidate_configs(
        hidden_dim_candidates,
        lr_candidates,
        weight_decay_candidates,
    )

    def eval_candidate_fn(*, config, idx_train, idx_val, fold_id, split_meta):
        del split_meta
        res = _fit_predict_inverse_kan(
            X_dev_raw[idx_train],
            y_dev_raw[idx_train],
            X_dev_raw[idx_val],
            hidden_dim=config["hidden_dim"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            epochs=epochs,
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
        model_name="inverse_KAN",
        task_name="inverse",
    )
    best_config = tuning_result["best_config"]
    best_summary = tuning_result["candidate_summaries"][tuning_result["best_candidate_idx"]]

    final_fit = _fit_predict_inverse_kan(
        X_dev_raw,
        y_dev_raw,
        X_test_raw,
        hidden_dim=best_config["hidden_dim"],
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
        epochs=epochs,
        device=device,
        seed=int(tuning_config.seed),
        return_model=save_artifacts,
    )
    model_final = final_fit["model"]
    pred_test_speed = final_fit["y_pred_eval"]

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_tracking = build_sample_tracking_columns(sample_meta, idx_te)
    split_indices_payload = build_single_split_artifact_payload(
        idx_tr,
        idx_val,
        idx_te,
        n_samples=len(X_raw),
    )

    r2_all = _safe_r2(y_test_speed_true, pred_test_speed)
    are_all = _safe_are(y_test_speed_true, pred_test_speed)
    n_all = int(len(y_test_speed_true))

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
        y_true_main = y_test_speed_true[policy_mask]
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
            **test_tracking,
            "target_mass_g_min": test_mass,
            "actual_opening_mm": test_opening,
            "strategy_opening_mm": strategy_opening_all,
            "true_speed_r_min": y_test_speed_true,
            "inverse_KAN_pred": pred_test_speed,
            "policy_match": policy_mask.astype(int),
        })
        save_dataframe(df_all, os.path.join(save_outputs_dir, "inverse_kan_predictions_all.csv"))

        df_main = pd.DataFrame({
            **build_sample_tracking_columns(sample_meta, idx_te[policy_mask]),
            "target_mass_g_min": mass_main,
            "actual_opening_mm": opening_main,
            "strategy_opening_mm": strategy_opening_main,
            "true_speed_r_min": y_true_main,
            "inverse_KAN_pred": y_pred_main,
        })
        save_dataframe(df_main, os.path.join(save_outputs_dir, "inverse_kan_predictions_main.csv"))

    if save_tuning_records_path is not None:
        save_dataframe(pd.DataFrame(tuning_result["tuning_records"]), save_tuning_records_path)

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
        artifact_test_inputs_path, artifact_test_targets_path = save_inverse_artifacts(
            model_final,
            model_path=model_path,
            meta_path=meta_path,
            data_path=data_path,
            best_config={
                "hidden_dim": int(best_config["hidden_dim"]),
                "lr": float(best_config["lr"]),
                "weight_decay": float(best_config["weight_decay"]),
                "epochs": int(epochs),
                "best_val_r2": float(best_summary["mean_val_r2"]),
            },
            norm_stats=final_fit["norm_stats"],
            x_train_domain_raw=X_dev_raw,
            y_train_domain_raw=y_dev_raw,
            idx_tr=idx_tr,
            idx_val=idx_val,
            idx_te=idx_te,
            tuning_config=tuning_config,
            artifact_extra=artifact_extra,
            artifact_source_files=artifact_source_files,
            x_test_raw=X_test_raw,
            y_test_raw=y_test_speed_true,
            save_test_slice_flag=save_test_slice,
        )

    _cleanup_torch_runtime(model=model_final, device=device)

    return {
        "r2_main": float(r2_main) if not np.isnan(r2_main) else np.nan,
        "are_main": float(are_main) if not np.isnan(are_main) else np.nan,
        "n_main": n_main,
        "main_ratio": float(main_ratio) if not np.isnan(main_ratio) else np.nan,
        "r2_all": float(r2_all) if not np.isnan(r2_all) else np.nan,
        "are_all": float(are_all) if not np.isnan(are_all) else np.nan,
        "n_all": n_all,
        "best_hidden_dim": int(best_config["hidden_dim"]),
        "best_lr": float(best_config["lr"]),
        "best_weight_decay": float(best_config["weight_decay"]),
        "best_config": {
            "hidden_dim": int(best_config["hidden_dim"]),
            "lr": float(best_config["lr"]),
            "weight_decay": float(best_config["weight_decay"]),
        },
        "sample_id_all": test_tracking["sample_id"],
        "source_row_number_all": test_tracking["source_row_number"],
        "y_true_all": np.asarray(y_test_speed_true),
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
        "artifact_dir": artifact_dir,
        "artifact_model_path": model_path,
        "artifact_meta_path": meta_path,
        "artifact_test_inputs_path": artifact_test_inputs_path,
        "artifact_test_targets_path": artifact_test_targets_path,
        "tuning_protocol": tuning_config_to_dict(tuning_config),
        "trial_budget": int(tuning_config.n_candidates),
        "validation_repeats": int(tuning_config.n_repeats),
        "selection_metric": tuning_config.selection_metric,
        "tie_break_metric": tuning_config.tie_break_metric,
        "tuning_records": tuning_result["tuning_records"],
        "candidate_summaries": tuning_result["candidate_summaries"],
        "norm_stats": final_fit["norm_stats"],
        "split_indices_payload": split_indices_payload,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the inverse KAN baseline.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = int(args.seed)
    run_dir = create_run_dir("inverse_kan")
    artifact_dir = os.path.join(run_dir, "artifacts", "inverse", "inverse_KAN")
    tuning_csv = os.path.join(run_dir, "tuning_records_inverse_kan.csv")
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
        script_name="inverse_kan.py",
        data_path=data_path,
        seed=seed,
        params={
            "hidden_dim_candidates": DEFAULT_HIDDEN_DIM_CANDIDATES,
            "lr_candidates": DEFAULT_LR_CANDIDATES,
            "weight_decay_candidates": DEFAULT_WEIGHT_DECAY_CANDIDATES,
            "epochs": 1000,
            "policy": {
                "label": POLICY_LABEL,
                "target_openings_mm": list(POLICY_TARGET_OPENINGS),
                "threshold_low_mid": POLICY_LOW_MID_THRESHOLD,
                "threshold_mid_high": POLICY_MID_HIGH_THRESHOLD,
            },
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

    res = train_and_eval_inverse_kan_v2(
        data_path=data_path,
        seed=seed,
        save_outputs_dir=run_dir,
        split_indices=split_indices,
        save_artifacts=True,
        artifact_dir=artifact_dir,
        save_tuning_records_path=tuning_csv,
        save_test_slice=True,
        artifact_extra={
            "run_dir": run_dir.replace("\\", "/"),
            "reference_output": {
                "path": "inverse_kan_predictions_all.csv",
                "prediction_column": "inverse_KAN_pred",
                "target_column": "true_speed_r_min",
            },
        },
    )

    update_manifest_split_artifact(run_dir, split_payload=res["split_indices_payload"])

    outputs = [
        {"path": "inverse_kan_predictions_all.csv"},
        {"path": "inverse_kan_predictions_main.csv"},
        {"path": "tuning_records_inverse_kan.csv"},
    ]
    if res["artifact_model_path"] is not None:
        outputs.append({"path": "artifacts/inverse/inverse_KAN/model.pth"})
    if res["artifact_meta_path"] is not None:
        outputs.append({"path": "artifacts/inverse/inverse_KAN/meta.json"})
    if res["artifact_test_inputs_path"] is not None:
        outputs.append({"path": "artifacts/inverse/inverse_KAN/test_inputs.npy"})
    if res["artifact_test_targets_path"] is not None:
        outputs.append({"path": "artifacts/inverse/inverse_KAN/test_targets.npy"})

    append_manifest_outputs(run_dir, outputs)
    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()
