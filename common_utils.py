# -*- coding: utf-8 -*-
"""
公共工具：
- 数据加载
- train/val/test 划分（联合分层增强版，含二次稀有类修复）
- 指标函数
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EPS = 1e-8


def average_relative_error(y_true, y_pred, eps=EPS):
    """
    平均相对误差（ARE, 百分比）
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")

    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    re = np.abs((y_pred - y_true) / denom) * 100.0
    return float(np.mean(re))


def load_data(path="data/dataset.xlsx"):
    """
    读取数据，返回 X, y
    X: (n_samples, 2) -> [排肥口开度（mm）, 排肥轴转速（r/min）]
    y: (n_samples,)   -> 实际排肥质量（g/min）
    """
    df = pd.read_excel(path)

    required_cols = ["排肥口开度（mm）", "排肥轴转速（r/min）", "实际排肥质量（g/min）"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少必要列：{col}")

    X = df[["排肥口开度（mm）", "排肥轴转速（r/min）"]].values.astype(np.float32)
    y = df["实际排肥质量（g/min）"].values.astype(np.float32)

    if len(X) == 0:
        raise ValueError("数据为空，无法进行训练/评估")

    return X, y


def _bin_by_quantiles(values, n_bins=3):
    """
    按分位数离散化。若唯一值过少，则自动降低 bin 数。
    """
    values = np.asarray(values).reshape(-1)

    unique_vals = np.unique(values)
    unique_count = len(unique_vals)

    if unique_count <= 1:
        return np.zeros(len(values), dtype=int)

    n_bins = min(max(2, int(n_bins)), unique_count)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)

    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)

    inner_edges = edges[1:-1]
    bins = np.digitize(values, inner_edges, right=False)
    return bins.astype(int)


def _build_joint_stratify_labels(X, y, opening_bins=3, speed_bins=3, mass_bins=3):
    """
    构造联合分层标签：
        label = opening_bin + speed_bin + mass_bin
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    opening = X[:, 0]
    speed = X[:, 1]
    mass = y

    opening_b = _bin_by_quantiles(opening, n_bins=opening_bins)
    speed_b = _bin_by_quantiles(speed, n_bins=speed_bins)
    mass_b = _bin_by_quantiles(mass, n_bins=mass_bins)

    labels = np.array(
        [f"{o}_{s}_{m}" for o, s, m in zip(opening_b, speed_b, mass_b)],
        dtype=object
    )
    return labels


def _merge_rare_classes(labels, min_count=2):
    """
    将过少样本类别并入 RARE，避免 stratify 因类别样本过少而失败。
    """
    labels = np.asarray(labels, dtype=object)
    unique, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(unique, counts))

    merged = np.array(
        [lab if count_map[lab] >= min_count else "RARE" for lab in labels],
        dtype=object
    )
    return merged


def _is_valid_stratify_labels(labels, min_count=2):
    """
    判断一组 labels 是否适合直接传给 train_test_split(..., stratify=labels)
    """
    if labels is None:
        return False

    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0:
        return False

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) <= 1:
        return False

    return np.min(counts) >= min_count


def get_train_val_test_indices(
    n_samples=None,
    X=None,
    y=None,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    use_stratify=True,
):
    """
    统一 train/val/test 三分数据，保证三个模型使用完全相同的样本索引。

    推荐新调用方式：
        idx_train, idx_val, idx_test = get_train_val_test_indices(X=X, y=y)

    兼容旧调用方式：
        idx_train, idx_val, idx_test = get_train_val_test_indices(n_samples=len(X))

    默认：
        train 70%, val 15%, test 15%

    说明：
    - 若提供 X 和 y，默认执行“开度 + 转速 + 排肥量”的联合分层切分
    - 若分层条件不足，则自动退回普通随机切分
    - 第二次切分前会再次合并稀有类，避免二次 stratify 报错
    """
    if X is None or y is None:
        if n_samples is None:
            raise ValueError("必须提供 n_samples，或同时提供 X 和 y")

        idx_all = np.arange(int(n_samples))

        idx_train_val, idx_test = train_test_split(
            idx_all,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        val_size_rel = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_size_rel,
            random_state=random_state,
            shuffle=True,
        )

        return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if len(X) != len(y):
        raise ValueError("X 与 y 长度不一致")
    if len(X) == 0:
        raise ValueError("数据为空")
    if n_samples is not None and int(n_samples) != len(X):
        raise ValueError("n_samples 与 len(X) 不一致")

    idx_all = np.arange(len(X))

    stratify_labels = None
    if use_stratify:
        try:
            stratify_labels = _build_joint_stratify_labels(X, y)
            stratify_labels = _merge_rare_classes(stratify_labels, min_count=2)

            if not _is_valid_stratify_labels(stratify_labels, min_count=2):
                stratify_labels = None
        except Exception:
            stratify_labels = None

    # ---------- 第一次切分：train_val / test ----------
    if stratify_labels is not None:
        idx_train_val, idx_test, lab_train_val, _ = train_test_split(
            idx_all,
            stratify_labels,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
    else:
        idx_train_val, idx_test = train_test_split(
            idx_all,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        lab_train_val = None

    # ---------- 第二次切分：train / val ----------
    val_size_rel = val_size / (1.0 - test_size)

    if lab_train_val is not None:
        # 关键修复：第一次切分后，类别数可能再次变稀疏，必须重新合并
        lab_train_val = _merge_rare_classes(lab_train_val, min_count=2)

        if not _is_valid_stratify_labels(lab_train_val, min_count=2):
            lab_train_val = None

    if lab_train_val is not None:
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_size_rel,
            random_state=random_state,
            shuffle=True,
            stratify=lab_train_val,
        )
    else:
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_size_rel,
            random_state=random_state,
            shuffle=True,
        )

    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)