# -*- coding: utf-8 -*-
"""
run_utils.py

Shared helpers for run directories, manifests, artifact bundles, and replay metadata.
"""

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1
RUN_SPLIT_SCHEMA_VERSION = 1
SAMPLE_TRACKING_INFO = {
    "sample_id_definition": (
        "0-based row index after loading the original input table; "
        "stable only within the current input file, not a cross-dataset global key"
    ),
    "source_row_number_definition": (
        "1-based visible spreadsheet row number from the original input table, "
        "including header offset"
    ),
}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _normalize_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def _repo_base_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _relativize_path(path: str, *, base_dir: str = None) -> str:
    abs_path = os.path.abspath(path)
    base_dir = os.path.abspath(base_dir or _repo_base_dir())
    try:
        rel_path = os.path.relpath(abs_path, base_dir)
        if not rel_path.startswith(".."):
            return rel_path.replace("\\", "/")
    except Exception:
        pass
    return abs_path.replace("\\", "/")


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, payload):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
    return path


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def get_env_info():
    env = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import sklearn
        env["sklearn_version"] = sklearn.__version__
    except Exception:
        env["sklearn_version"] = None

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        env["torch_version"] = None
        env["torch_cuda_available"] = None

    try:
        import pandas
        env["pandas_version"] = pandas.__version__
    except Exception:
        env["pandas_version"] = None

    try:
        import numpy
        env["numpy_version"] = numpy.__version__
    except Exception:
        env["numpy_version"] = None

    return env


def _run_git_command(args, cwd):
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def get_code_version(source_files=None):
    repo_dir = _repo_base_dir()
    normalized_files = []
    for path in list(source_files or []):
        if path is None:
            continue
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            normalized_files.append(abs_path)

    source_file_hashes = {}
    for abs_path in sorted(set(normalized_files)):
        source_file_hashes[_relativize_path(abs_path, base_dir=repo_dir)] = sha256_of_file(abs_path)

    git_commit = None
    git_branch = None
    git_dirty = None

    inside_work_tree = _run_git_command(["git", "rev-parse", "--is-inside-work-tree"], repo_dir)
    if inside_work_tree == "true":
        git_commit = _run_git_command(["git", "rev-parse", "HEAD"], repo_dir)
        git_branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
        status = _run_git_command(["git", "status", "--porcelain"], repo_dir)
        if status is not None:
            git_dirty = bool(status.strip())

    return {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "source_file_hashes": source_file_hashes,
    }


def build_data_fingerprint(data_path: str):
    abs_path = os.path.abspath(data_path)
    exists = os.path.exists(abs_path)
    return {
        "path": _normalize_path(abs_path),
        "sha256": sha256_of_file(abs_path) if exists else None,
    }


def get_command_info():
    return {
        "argv": [str(arg) for arg in sys.argv],
        "cwd": _normalize_path(os.getcwd()),
        "python_executable": _normalize_path(sys.executable),
    }


def create_run_dir(entrypoint: str, runs_root: str = "runs") -> str:
    ts = now_timestamp()
    run_dir = os.path.join(runs_root, f"{ts}_{entrypoint}")
    ensure_dir(run_dir)
    return run_dir


def build_split_indices_payload(idx_train, idx_val, idx_test):
    return {
        "idx_train": np.asarray(idx_train, dtype=int).reshape(-1),
        "idx_val": np.asarray(idx_val, dtype=int).reshape(-1),
        "idx_test": np.asarray(idx_test, dtype=int).reshape(-1),
    }


def build_single_split_artifact_payload(idx_train, idx_val, idx_test, *, n_samples, extra=None):
    payload = {
        "schema_version": int(RUN_SPLIT_SCHEMA_VERSION),
        "kind": "single_split",
        "n_samples": int(n_samples),
        "fold_count": 1,
    }
    payload.update(build_split_indices_payload(idx_train, idx_val, idx_test))
    if extra:
        payload.update(jsonable(extra))
    return payload


def build_multi_fold_split_artifact_payload(folds, *, n_samples):
    serialized_folds = []
    for fold in list(folds or []):
        payload = {
            "fold_id": int(fold["fold_id"]),
            "protocol": fold["protocol"],
            "idx_train": np.asarray(fold["idx_train"], dtype=int).reshape(-1),
            "idx_val": np.asarray(fold["idx_val"], dtype=int).reshape(-1),
            "idx_test": np.asarray(fold["idx_test"], dtype=int).reshape(-1),
        }
        for key, value in dict(fold).items():
            if key in payload:
                continue
            payload[key] = jsonable(value)
        serialized_folds.append(payload)

    return {
        "schema_version": int(RUN_SPLIT_SCHEMA_VERSION),
        "kind": "multi_fold",
        "n_samples": int(n_samples),
        "fold_count": len(serialized_folds),
        "folds": serialized_folds,
    }


def _split_indices_summary(run_dir: str, split_path: str, split_payload):
    kind = str(split_payload.get("kind", "single_split"))
    fold_count = split_payload.get("fold_count")
    if fold_count is None:
        fold_count = 1 if kind == "single_split" else len(split_payload.get("folds", []))
    return {
        "path": os.path.relpath(split_path, run_dir).replace("\\", "/"),
        "sha256": sha256_of_file(split_path),
        "kind": kind,
        "fold_count": int(fold_count),
        "n_samples": int(split_payload.get("n_samples", 0)),
    }


def write_split_indices_artifact(run_dir: str, split_payload, filename: str = "split_indices.json"):
    if not split_payload:
        return None

    payload = dict(jsonable(split_payload))
    payload.setdefault("schema_version", int(RUN_SPLIT_SCHEMA_VERSION))
    kind = str(payload.get("kind", ""))
    if kind not in {"single_split", "multi_fold"}:
        raise ValueError("split_payload.kind must be 'single_split' or 'multi_fold'")
    if kind == "single_split":
        payload["fold_count"] = 1
    else:
        payload["fold_count"] = int(payload.get("fold_count", len(payload.get("folds", []))))

    split_path = os.path.join(run_dir, filename)
    write_json(split_path, payload)
    return _split_indices_summary(run_dir, split_path, payload)


def build_tuning_protocol_payload(
    tuning_protocol,
    *,
    inner_split_strategy=None,
    inner_split_meta=None,
    inner_splits=None,
    tuning_seed=None,
    n_repeats=None,
    inner_val_ratio=None,
):
    payload = dict(jsonable(tuning_protocol or {}))
    payload["inner_split_strategy"] = inner_split_strategy
    payload["inner_split_meta"] = jsonable(inner_split_meta or {})
    payload["tuning_seed"] = tuning_seed
    payload["n_repeats"] = n_repeats
    payload["inner_val_ratio"] = inner_val_ratio
    if inner_splits is not None:
        payload["inner_splits"] = jsonable(inner_splits)
    return payload


def build_artifact_metadata(
    *,
    artifact_type,
    task_name,
    model_name,
    model_class,
    data_path,
    best_config,
    normalization_params,
    split_indices,
    tuning_protocol,
    training_domain,
    extra=None,
    source_files=None,
):
    return {
        "schema_version": int(ARTIFACT_SCHEMA_VERSION),
        "artifact_type": artifact_type,
        "task_name": task_name,
        "model_name": model_name,
        "model_class": model_class,
        "data": build_data_fingerprint(data_path),
        "code_version": get_code_version(source_files=source_files),
        "best_config": jsonable(best_config or {}),
        "normalization_params": jsonable(normalization_params or {}),
        "split_indices": jsonable(split_indices or {}),
        "tuning_protocol": jsonable(tuning_protocol or {}),
        "training_domain": jsonable(training_domain or {}),
        "extra": jsonable(extra or {}),
    }


def save_test_slice(
    artifact_dir,
    X_test,
    y_test,
    *,
    x_filename="test_inputs.npy",
    y_filename="test_targets.npy",
):
    ensure_dir(artifact_dir)
    x_path = os.path.join(artifact_dir, x_filename)
    y_path = os.path.join(artifact_dir, y_filename)
    np.save(x_path, np.asarray(X_test))
    np.save(y_path, np.asarray(y_test))
    return x_path, y_path


def write_manifest(
    run_dir: str,
    *,
    script_name: str,
    data_path: str,
    seed=None,
    params=None,
    extra=None,
    source_files=None,
    split_payload=None,
    split_artifact_summary=None,
):
    script_path = script_name
    if not os.path.isabs(script_path):
        script_path = os.path.join(_repo_base_dir(), script_name)

    resolved_source_files = list(source_files or [])
    resolved_source_files.append(script_path)
    resolved_source_files.append(__file__)

    manifest = {
        "script_name": script_name,
        "timestamp": os.path.basename(run_dir).split("_")[0],
        "run_dir": run_dir.replace("\\", "/"),
        "data": build_data_fingerprint(data_path),
        "seed": seed,
        "params": jsonable(params or {}),
        "command": get_command_info(),
        "environment": jsonable(get_env_info()),
        "code_version": get_code_version(source_files=resolved_source_files),
        "sample_tracking": dict(SAMPLE_TRACKING_INFO),
        "artifacts": {},
        "outputs": [],
        "extra": jsonable(extra or {}),
    }

    if split_payload is not None and split_artifact_summary is None:
        split_artifact_summary = write_split_indices_artifact(run_dir, split_payload)
    if split_artifact_summary is not None:
        manifest["artifacts"]["split_indices"] = jsonable(split_artifact_summary)

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    write_json(manifest_path, manifest)
    return manifest_path


def update_manifest_split_artifact(
    run_dir: str,
    *,
    split_payload=None,
    split_artifact_summary=None,
    filename: str = "split_indices.json",
):
    if split_payload is None and split_artifact_summary is None:
        return None

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if split_artifact_summary is None:
        split_artifact_summary = write_split_indices_artifact(
            run_dir,
            split_payload,
            filename=filename,
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts = dict(manifest.get("artifacts") or {})
    artifacts["split_indices"] = jsonable(split_artifact_summary)
    manifest["artifacts"] = artifacts
    manifest.setdefault("command", get_command_info())
    manifest.setdefault("sample_tracking", dict(SAMPLE_TRACKING_INFO))
    write_json(manifest_path, manifest)
    return split_artifact_summary


def append_manifest_outputs(run_dir: str, outputs):
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"未找到 manifest：{manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    existing = list(manifest.get("outputs", []))
    for item in outputs:
        item = jsonable(item)
        if item is None:
            continue
        if item not in existing:
            existing.append(item)

    manifest["outputs"] = existing
    write_json(manifest_path, manifest)


def save_dataframe(df: pd.DataFrame, path: str):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def compare_csv_file(
    current_path: str,
    baseline_path: str,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
):
    if not os.path.exists(current_path):
        return False, f"当前文件不存在：{current_path}"
    if not os.path.exists(baseline_path):
        return False, f"基线文件不存在：{baseline_path}"

    df_cur = pd.read_csv(current_path)
    df_base = pd.read_csv(baseline_path)

    if list(df_cur.columns) != list(df_base.columns):
        return False, (
            "列名不一致：\n"
            f"current={list(df_cur.columns)}\n"
            f"baseline={list(df_base.columns)}"
        )

    if df_cur.shape != df_base.shape:
        return False, f"形状不一致：current={df_cur.shape}, baseline={df_base.shape}"

    for col in df_cur.columns:
        s1 = df_cur[col]
        s2 = df_base[col]

        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            a = s1.to_numpy(dtype=float)
            b = s2.to_numpy(dtype=float)
            same = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
            if not np.all(same):
                idx = int(np.where(~same)[0][0])
                return False, (
                    f"数值列 `{col}` 不一致，首个差异在第 {idx} 行："
                    f"current={a[idx]}, baseline={b[idx]}"
                )
        else:
            a = s1.astype(str).tolist()
            b = s2.astype(str).tolist()
            if a != b:
                idx = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                return False, (
                    f"文本列 `{col}` 不一致，首个差异在第 {idx} 行："
                    f"current={a[idx]!r}, baseline={b[idx]!r}"
                )

    return True, "一致"
