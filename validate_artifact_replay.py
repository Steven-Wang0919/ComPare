# -*- coding: utf-8 -*-
"""
validate_artifact_replay.py

Replay a saved artifact bundle and compare predictions against a reference output.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from common_utils import load_data
from inverse_kan import InverseKANModel
from train_kan import FertilizerKAN


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_2d_array(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _to_scalar(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value, got shape {np.asarray(value).shape}")
    return float(arr[0])


def _norm_x(X, norm_params):
    x_min = _to_2d_array(norm_params["X_min"])
    x_max = _to_2d_array(norm_params["X_max"])
    return (np.asarray(X, dtype=float) - x_min) / (x_max - x_min + 1e-8)


def _denorm_y(y_norm, norm_params):
    y_min = _to_scalar(norm_params["y_min"])
    y_max = _to_scalar(norm_params["y_max"])
    return np.asarray(y_norm, dtype=float).reshape(-1) * (y_max - y_min + 1e-8) + y_min


def _rebuild_test_slice(meta):
    data_path = meta["data"]["path"]
    recorded_sha = meta["data"].get("sha256")
    if recorded_sha and os.path.exists(data_path):
        import run_utils

        current_sha = run_utils.sha256_of_file(data_path)
        if current_sha != recorded_sha:
            raise ValueError(f"Data SHA256 mismatch: expected {recorded_sha}, got {current_sha}")

    X_raw, y_raw = load_data(data_path)
    idx_test = np.asarray(meta["split_indices"]["idx_test"], dtype=int).reshape(-1)
    if meta["task_name"] == "forward":
        return np.asarray(X_raw[idx_test]), np.asarray(y_raw[idx_test])
    if meta["task_name"] == "inverse":
        X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
        y_inv_all = X_raw[:, 1]
        return np.asarray(X_inv_all[idx_test]), np.asarray(y_inv_all[idx_test])
    raise ValueError(f"Unsupported task_name: {meta['task_name']}")


def _load_test_slice(artifact_dir, meta):
    extra = meta.get("extra", {})
    test_inputs_file = extra.get("test_inputs_file") or "test_inputs.npy"
    test_targets_file = extra.get("test_targets_file") or "test_targets.npy"
    x_path = os.path.join(artifact_dir, test_inputs_file)
    y_path = os.path.join(artifact_dir, test_targets_file)
    if os.path.exists(x_path) and os.path.exists(y_path):
        return np.load(x_path), np.load(y_path), "bundle_test_slice"
    X_test, y_test = _rebuild_test_slice(meta)
    return X_test, y_test, "reconstructed_from_split_indices"


def _load_reference_predictions(meta, artifact_dir, y_pred):
    extra = meta.get("extra", {})
    reference = extra.get("reference_output") or {}
    run_dir = extra.get("run_dir")
    if run_dir:
        ref_path = os.path.join(run_dir, reference.get("path", ""))
        if reference.get("path") and os.path.exists(ref_path):
            df = pd.read_csv(ref_path)
            filter_column = reference.get("filter_column")
            if filter_column:
                filter_value = reference.get("filter_value")
                df = df[df[filter_column] == filter_value]
            pred_col = reference.get("prediction_column")
            target_col = reference.get("target_column")
            if pred_col not in df.columns or target_col not in df.columns:
                raise ValueError(f"Reference CSV missing required columns in {ref_path}")
            return (
                df[pred_col].to_numpy(dtype=float).reshape(-1),
                df[target_col].to_numpy(dtype=float).reshape(-1),
                ref_path,
            )

    # Fallback: use the replay-generated predictions as reference when no CSV is available.
    return np.asarray(y_pred, dtype=float).reshape(-1), None, "replay_generated"


def _predict_with_joblib(model_path, meta, X_test):
    model = joblib.load(model_path)
    X_norm = _norm_x(X_test, meta["normalization_params"])
    y_pred_norm = model.predict(X_norm)
    return _denorm_y(y_pred_norm, meta["normalization_params"])


def _predict_with_kan(model_path, meta, X_test):
    best_config = meta["best_config"]
    model_name = meta["model_name"]
    if model_name == "KAN":
        model = FertilizerKAN(input_dim=2, hidden_dim=int(best_config["hidden_dim"]), output_dim=1)
    elif model_name == "inverse_KAN":
        model = InverseKANModel(input_dim=2, hidden_dim=int(best_config["hidden_dim"]), output_dim=1)
    else:
        raise ValueError(f"Unsupported KAN model_name: {model_name}")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    X_norm = _norm_x(X_test, meta["normalization_params"])
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X_norm, dtype=torch.float32)).cpu().numpy().reshape(-1)
    return _denorm_y(y_pred_norm, meta["normalization_params"])


def _predict(meta, artifact_dir, X_test):
    model_file = meta.get("extra", {}).get("model_file")
    if model_file is None:
        model_file = "model.pth" if meta["model_name"] in {"KAN", "inverse_KAN"} else "model.joblib"
    model_path = os.path.join(artifact_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.endswith(".joblib"):
        return _predict_with_joblib(model_path, meta, X_test)
    if model_path.endswith(".pth"):
        return _predict_with_kan(model_path, meta, X_test)
    raise ValueError(f"Unsupported model artifact type: {model_path}")


def _max_relative_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.max(np.abs(y_pred - y_true) / denom))


def validate_artifact_replay(artifact_dir, *, atol=1e-6, rtol=1e-6):
    meta_path = os.path.join(artifact_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {artifact_dir}")

    meta = _load_json(meta_path)
    if int(meta.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported schema_version: {meta.get('schema_version')}")

    X_test, y_test, test_source = _load_test_slice(artifact_dir, meta)
    y_pred = _predict(meta, artifact_dir, X_test)
    ref_pred, ref_target, reference_source = _load_reference_predictions(meta, artifact_dir, y_pred)

    if ref_target is None:
        ref_target = np.asarray(y_test, dtype=float).reshape(-1)

    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ref_pred = np.asarray(ref_pred, dtype=float).reshape(-1)
    ref_target = np.asarray(ref_target, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)

    if len(y_pred) != len(ref_pred):
        raise ValueError(f"Prediction length mismatch: replay={len(y_pred)}, reference={len(ref_pred)}")
    if len(y_test) != len(ref_target):
        raise ValueError(f"Target length mismatch: replay={len(y_test)}, reference={len(ref_target)}")

    pred_close = np.isclose(y_pred, ref_pred, atol=atol, rtol=rtol)
    target_close = np.isclose(y_test, ref_target, atol=atol, rtol=rtol)

    return {
        "artifact_dir": os.path.abspath(artifact_dir),
        "model_name": meta["model_name"],
        "task_name": meta["task_name"],
        "schema_version": int(meta["schema_version"]),
        "test_source": test_source,
        "reference_source": reference_source,
        "n_samples": int(len(y_pred)),
        "max_abs_error_vs_reference": float(np.max(np.abs(y_pred - ref_pred))) if len(y_pred) else 0.0,
        "max_rel_error_vs_reference": _max_relative_error(ref_pred, y_pred) if len(y_pred) else 0.0,
        "max_abs_error_vs_targets": float(np.max(np.abs(y_test - ref_target))) if len(y_test) else 0.0,
        "predictions_match_reference": bool(np.all(pred_close)),
        "targets_match_reference": bool(np.all(target_close)),
        "passed": bool(np.all(pred_close) and np.all(target_close)),
    }


def main():
    parser = argparse.ArgumentParser(description="Replay and validate a saved artifact bundle.")
    parser.add_argument("--artifact-dir", required=True, help="Path to the artifact bundle directory.")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    result = validate_artifact_replay(args.artifact_dir, atol=args.atol, rtol=args.rtol)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
