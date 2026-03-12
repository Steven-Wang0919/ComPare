# -*- coding: utf-8 -*-
"""
run_utils.py

统一管理运行目录、manifest、数据校验和、环境信息与 CSV 输出。
"""

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def create_run_dir(entrypoint: str, runs_root: str = "runs") -> str:
    ts = now_timestamp()
    run_dir = os.path.join(runs_root, f"{ts}_{entrypoint}")
    ensure_dir(run_dir)
    return run_dir


def write_manifest(
    run_dir: str,
    *,
    script_name: str,
    data_path: str,
    seed=None,
    params=None,
    extra=None,
):
    manifest = {
        "script_name": script_name,
        "timestamp": os.path.basename(run_dir).split("_")[0],
        "run_dir": run_dir.replace("\\", "/"),
        "data": {
            "path": data_path.replace("\\", "/"),
            "sha256": sha256_of_file(data_path) if os.path.exists(data_path) else None,
        },
        "seed": seed,
        "params": jsonable(params or {}),
        "environment": jsonable(get_env_info()),
        "outputs": [],
        "extra": jsonable(extra or {}),
    }

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path


def append_manifest_outputs(run_dir: str, outputs):
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"未找到 manifest：{manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    existing = list(manifest.get("outputs", []))
    for item in outputs:
        item = jsonable(item)
        if item not in existing:
            existing.append(item)

    manifest["outputs"] = existing

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


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