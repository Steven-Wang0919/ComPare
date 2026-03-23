# -*- coding: utf-8 -*-
"""
公共工具：
- 数据加载
- train/val/test 划分（联合分层增强版，含二次稀有类修复）
- 可解释评估协议构建（随机插值 / 留一开度外推 / 留一速度段外推）
- 指标函数
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EPS = 1e-8
DEFAULT_TARGET_OPENINGS = (20.0, 35.0, 50.0)
DEFAULT_OPENING_ATOL = 0.1
DEFAULT_STRATIFY_VIEW = "forward"
DEFAULT_OPENING_BINS = 3
DEFAULT_SPEED_BINS = 3
DEFAULT_MASS_BINS = 3
DEFAULT_RARE_CLASS_MERGE_MIN_COUNT = 2
DEFAULT_RARE_CLASS_PLACEHOLDER = "RARE"
SUPPORTED_STRATIFY_VIEWS = ("forward", "inverse", "physical_joint")


def average_relative_error(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    re = np.abs((y_pred - y_true) / denom) * 100.0
    return float(np.mean(re))


def load_data_with_metadata(path="data/dataset.xlsx"):
    df = pd.read_excel(path)
    required_cols = ["排肥口开度（mm）", "排肥轴转速（r/min）", "实际排肥质量（g/min）"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少必要列：{col}")
    X = df[["排肥口开度（mm）", "排肥轴转速（r/min）"]].values.astype(np.float32)
    y = df["实际排肥质量（g/min）"].values.astype(np.float32)
    if len(X) == 0:
        raise ValueError("数据为空，无法进行训练/评估")
    sample_id = np.arange(len(df), dtype=int)
    sample_meta = pd.DataFrame({
        "sample_id": sample_id,
        "source_row_number": sample_id + 2,
    })
    return X, y, sample_meta


def load_data(path="data/dataset.xlsx"):
    X, y, _ = load_data_with_metadata(path)
    return X, y


def build_sample_tracking_columns(sample_meta, indices, *, include_legacy_sample_index=False):
    sample_meta_df = pd.DataFrame(sample_meta).reset_index(drop=True)
    if "sample_id" not in sample_meta_df.columns or "source_row_number" not in sample_meta_df.columns:
        raise ValueError("sample_meta must include sample_id and source_row_number")

    idx = np.asarray(indices, dtype=int).reshape(-1)
    selected = sample_meta_df.iloc[idx]
    payload = {
        "sample_id": selected["sample_id"].to_numpy(dtype=int),
        "source_row_number": selected["source_row_number"].to_numpy(dtype=int),
    }
    if include_legacy_sample_index:
        payload["sample_index"] = payload["sample_id"].copy()
    return payload


def _bin_by_quantiles(values, n_bins=3):
    values = np.asarray(values).reshape(-1)
    unique_count = len(np.unique(values))
    if unique_count <= 1:
        return np.zeros(len(values), dtype=int)
    n_bins = min(max(2, int(n_bins)), unique_count)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, edges[1:-1], right=False).astype(int)


def _normalize_stratify_view(stratify_view):
    view = str(stratify_view or DEFAULT_STRATIFY_VIEW).strip()
    if view not in SUPPORTED_STRATIFY_VIEWS:
        raise ValueError(
            f"Unsupported stratify_view={stratify_view!r}; "
            f"expected one of {SUPPORTED_STRATIFY_VIEWS}"
        )
    return view


def get_stratify_variable_mapping(stratify_view=DEFAULT_STRATIFY_VIEW):
    view = _normalize_stratify_view(stratify_view)
    if view == "forward":
        return {"opening": "X[:,0]", "speed": "X[:,1]", "mass": "y"}
    if view == "inverse":
        return {"opening": "X[:,1]", "speed": "y", "mass": "X[:,0]"}
    return {"opening": "raw.opening", "speed": "raw.speed", "mass": "raw.mass"}


def get_stratify_metadata(stratify_view=DEFAULT_STRATIFY_VIEW):
    return {
        "stratify_view": _normalize_stratify_view(stratify_view),
        "stratify_variable_mapping": get_stratify_variable_mapping(stratify_view),
        "opening_bins": int(DEFAULT_OPENING_BINS),
        "speed_bins": int(DEFAULT_SPEED_BINS),
        "mass_bins": int(DEFAULT_MASS_BINS),
        "rare_class_merge_min_count": int(DEFAULT_RARE_CLASS_MERGE_MIN_COUNT),
        "rare_class_placeholder": DEFAULT_RARE_CLASS_PLACEHOLDER,
    }


def _validate_stratify_arrays(opening, speed, mass):
    opening = np.asarray(opening, dtype=float).reshape(-1)
    speed = np.asarray(speed, dtype=float).reshape(-1)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    n = len(opening)
    if len(speed) != n or len(mass) != n:
        raise ValueError("opening/speed/mass must have the same number of samples.")
    return {
        "opening": opening,
        "speed": speed,
        "mass": mass,
    }


def _resolve_named_stratify_variables(
    X,
    y,
    *,
    stratify_view=DEFAULT_STRATIFY_VIEW,
    raw_X=None,
    raw_y=None,
):
    view = _normalize_stratify_view(stratify_view)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X must be a 2D array with at least two columns.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if view == "forward":
        return _validate_stratify_arrays(X[:, 0], X[:, 1], y)
    if view == "inverse":
        return _validate_stratify_arrays(X[:, 1], y, X[:, 0])

    src_X = np.asarray(raw_X if raw_X is not None else X, dtype=float)
    src_y = np.asarray(raw_y if raw_y is not None else y, dtype=float).reshape(-1)
    if src_X.ndim != 2 or src_X.shape[1] < 2:
        raise ValueError("raw_X must be a 2D array with at least two columns for physical_joint.")
    if len(src_X) != len(src_y):
        raise ValueError("raw_X and raw_y must contain the same number of samples.")
    return _validate_stratify_arrays(src_X[:, 0], src_X[:, 1], src_y)


def _build_joint_stratify_labels(
    named_variables,
    opening_bins=DEFAULT_OPENING_BINS,
    speed_bins=DEFAULT_SPEED_BINS,
    mass_bins=DEFAULT_MASS_BINS,
):
    vars_map = _validate_stratify_arrays(
        named_variables["opening"],
        named_variables["speed"],
        named_variables["mass"],
    )
    opening_b = _bin_by_quantiles(vars_map["opening"], n_bins=opening_bins)
    speed_b = _bin_by_quantiles(vars_map["speed"], n_bins=speed_bins)
    mass_b = _bin_by_quantiles(vars_map["mass"], n_bins=mass_bins)
    return np.array([f"{o}_{s}_{m}" for o, s, m in zip(opening_b, speed_b, mass_b)], dtype=object)


def build_joint_stratify_labels_for_view(
    X,
    y,
    *,
    stratify_view=DEFAULT_STRATIFY_VIEW,
    raw_X=None,
    raw_y=None,
):
    named_variables = _resolve_named_stratify_variables(
        X,
        y,
        stratify_view=stratify_view,
        raw_X=raw_X,
        raw_y=raw_y,
    )
    return _build_joint_stratify_labels(named_variables)


def _merge_rare_classes(
    labels,
    min_count=DEFAULT_RARE_CLASS_MERGE_MIN_COUNT,
    rare_placeholder=DEFAULT_RARE_CLASS_PLACEHOLDER,
):
    labels = np.asarray(labels, dtype=object)
    unique, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(unique, counts))
    return np.array(
        [lab if count_map[lab] >= min_count else rare_placeholder for lab in labels],
        dtype=object,
    )


def _is_valid_stratify_labels(labels, min_count=DEFAULT_RARE_CLASS_MERGE_MIN_COUNT):
    if labels is None:
        return False
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0:
        return False
    unique, counts = np.unique(labels, return_counts=True)
    return len(unique) > 1 and np.min(counts) >= min_count


def validate_predefined_split_indices(n_samples, idx_train, idx_val, idx_test):
    arrays = []
    for name, idx in zip(["train", "val", "test"], [idx_train, idx_val, idx_test]):
        arr = np.asarray(idx, dtype=int).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} 索引为空")
        if np.any(arr < 0) or np.any(arr >= int(n_samples)):
            raise ValueError(f"{name} 索引越界")
        if len(np.unique(arr)) != len(arr):
            raise ValueError(f"{name} 索引存在重复")
        arrays.append(np.sort(arr))
    train_set, val_set, test_set = map(set, arrays)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train/val/test 索引存在重叠")
    if len(train_set | val_set | test_set) != int(n_samples):
        raise ValueError("train/val/test 索引并集未覆盖全部样本")
    return arrays[0], arrays[1], arrays[2]


def _build_stratify_labels_or_none(
    X,
    y,
    use_stratify=True,
    *,
    stratify_view=DEFAULT_STRATIFY_VIEW,
    raw_X=None,
    raw_y=None,
):
    if not use_stratify:
        return None
    try:
        labels = build_joint_stratify_labels_for_view(
            X,
            y,
            stratify_view=stratify_view,
            raw_X=raw_X,
            raw_y=raw_y,
        )
        labels = _merge_rare_classes(
            labels,
            min_count=DEFAULT_RARE_CLASS_MERGE_MIN_COUNT,
            rare_placeholder=DEFAULT_RARE_CLASS_PLACEHOLDER,
        )
        if not _is_valid_stratify_labels(labels, min_count=DEFAULT_RARE_CLASS_MERGE_MIN_COUNT):
            return None
        return labels
    except Exception:
        return None


def get_train_val_test_indices(
    n_samples=None,
    X=None,
    y=None,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    use_stratify=True,
    stratify_view=DEFAULT_STRATIFY_VIEW,
    raw_X=None,
    raw_y=None,
):
    """
    统一 train/val/test 三分数据，保证三个模型使用完全相同的样本索引。

    当 test_size=0 时，返回 (train, val, empty_test)。
    """
    if X is None or y is None:
        if n_samples is None:
            raise ValueError("必须提供 n_samples，或同时提供 X 和 y")
        idx_all = np.arange(int(n_samples))
        if test_size == 0:
            idx_train, idx_val = train_test_split(
                idx_all,
                test_size=val_size,
                random_state=random_state,
                shuffle=True,
            )
            return np.sort(idx_train), np.sort(idx_val), np.array([], dtype=int)
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
    stratify_labels = _build_stratify_labels_or_none(
        X,
        y,
        use_stratify=use_stratify,
        stratify_view=stratify_view,
        raw_X=raw_X,
        raw_y=raw_y,
    )

    if test_size == 0:
        if not (0.0 < val_size < 1.0):
            raise ValueError("当 test_size=0 时，val_size 必须位于 (0,1) 区间")
        if stratify_labels is not None:
            idx_train, idx_val = train_test_split(
                idx_all,
                test_size=val_size,
                random_state=random_state,
                shuffle=True,
                stratify=stratify_labels,
            )
        else:
            idx_train, idx_val = train_test_split(
                idx_all,
                test_size=val_size,
                random_state=random_state,
                shuffle=True,
            )
        return np.sort(idx_train), np.sort(idx_val), np.array([], dtype=int)

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

    val_size_rel = val_size / (1.0 - test_size)
    if lab_train_val is not None:
        lab_train_val = _merge_rare_classes(
            lab_train_val,
            min_count=DEFAULT_RARE_CLASS_MERGE_MIN_COUNT,
            rare_placeholder=DEFAULT_RARE_CLASS_PLACEHOLDER,
        )
        if not _is_valid_stratify_labels(
            lab_train_val,
            min_count=DEFAULT_RARE_CLASS_MERGE_MIN_COUNT,
        ):
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


def make_split_from_masks(train_mask, val_mask, test_mask):
    train_mask = np.asarray(train_mask).astype(bool).reshape(-1)
    val_mask = np.asarray(val_mask).astype(bool).reshape(-1)
    test_mask = np.asarray(test_mask).astype(bool).reshape(-1)
    n = len(train_mask)
    if len(val_mask) != n or len(test_mask) != n:
        raise ValueError("train/val/test mask 长度不一致")
    return validate_predefined_split_indices(
        n,
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    )


def combine_train_val_indices(idx_train, idx_val):
    idx_train = np.asarray(idx_train, dtype=int).reshape(-1)
    idx_val = np.asarray(idx_val, dtype=int).reshape(-1)
    if idx_train.size == 0 or idx_val.size == 0:
        raise ValueError("train and val indices must both be non-empty")
    merged = np.concatenate([idx_train, idx_val])
    if len(np.unique(merged)) != len(merged):
        raise ValueError("train and val indices overlap")
    return np.sort(merged)


def is_target_opening(openings, target_openings=DEFAULT_TARGET_OPENINGS, atol=DEFAULT_OPENING_ATOL):
    openings = np.asarray(openings, dtype=float).reshape(-1)
    mask = np.zeros(len(openings), dtype=bool)
    for op in target_openings:
        mask |= np.isclose(openings, op, atol=atol)
    return mask


def build_opening_holdout_indices(
    X,
    y,
    random_state=42,
    val_ratio=0.2,
    target_openings=DEFAULT_TARGET_OPENINGS,
    atol=DEFAULT_OPENING_ATOL,
    stratify_view=DEFAULT_STRATIFY_VIEW,
):
    opening = np.asarray(X[:, 0], dtype=float).reshape(-1)
    test_mask = is_target_opening(opening, target_openings=target_openings, atol=atol)
    idx_test = np.where(test_mask)[0]
    idx_train_val = np.where(~test_mask)[0]

    if len(idx_test) == 0:
        raise ValueError("No target opening samples found for opening holdout protocol.")
    if len(idx_train_val) < 2:
        raise ValueError("Too few non-target-opening samples to form train/val splits.")

    idx_train_sub, idx_val_sub, _ = get_train_val_test_indices(
        X=np.asarray(X)[idx_train_val],
        y=np.asarray(y).reshape(-1)[idx_train_val],
        test_size=0.0,
        val_size=val_ratio,
        random_state=random_state,
        use_stratify=True,
        stratify_view=stratify_view,
    )
    idx_train = idx_train_val[idx_train_sub]
    idx_val = idx_train_val[idx_val_sub]
    return validate_predefined_split_indices(len(X), idx_train, idx_val, idx_test)


def build_protocol_splits(
    X,
    y,
    protocol="random_stratified",
    random_state=42,
    test_size=0.15,
    val_size=0.15,
    holdout_opening=None,
    holdout_speed_min=None,
    holdout_speed_max=None,
    val_opening=None,
    val_speed_min=None,
    val_speed_max=None,
    stratify_view=DEFAULT_STRATIFY_VIEW,
):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n_samples = len(X)
    if n_samples != len(y):
        raise ValueError("X 与 y 长度不一致")

    opening = X[:, 0]
    speed = X[:, 1]

    if protocol == "random_stratified":
        idx_train, idx_val, idx_test = get_train_val_test_indices(
            X=X,
            y=y,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            use_stratify=True,
            stratify_view=stratify_view,
        )
        return {
            "protocol": protocol,
            "description": "随机联合分层切分（规则网格内插值评估）",
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
            **get_stratify_metadata(stratify_view),
        }

    if protocol == "leave_one_opening_out":
        unique_openings = sorted(np.unique(opening).tolist())
        if holdout_opening is None:
            holdout_opening = unique_openings[-1]
        if holdout_opening not in unique_openings:
            raise ValueError(f"holdout_opening={holdout_opening} 不在可用开度层中：{unique_openings}")

        test_mask = np.isclose(opening, holdout_opening)
        remaining_idx = np.where(~test_mask)[0]
        if val_opening is not None:
            if val_opening == holdout_opening:
                raise ValueError("val_opening 不能与 holdout_opening 相同")
            val_mask = np.isclose(opening, val_opening)
            train_mask = ~(test_mask | val_mask)
            idx_train, idx_val, idx_test = make_split_from_masks(train_mask, val_mask, test_mask)
        else:
            idx_train_sub, idx_val_sub, _ = get_train_val_test_indices(
                X=X[remaining_idx],
                y=y[remaining_idx],
                test_size=0.0,
                val_size=val_size,
                random_state=random_state,
                use_stratify=True,
                stratify_view=stratify_view,
            )
            idx_train = remaining_idx[idx_train_sub]
            idx_val = remaining_idx[idx_val_sub]
            idx_test = np.where(test_mask)[0]
            idx_train, idx_val, idx_test = validate_predefined_split_indices(n_samples, idx_train, idx_val, idx_test)
        return {
            "protocol": protocol,
            "description": f"留一开度层外推：test 开度={holdout_opening}",
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
            "holdout_opening": float(holdout_opening),
            "val_opening": None if val_opening is None else float(val_opening),
            **get_stratify_metadata(stratify_view),
        }

    if protocol == "leave_speed_block_out":
        unique_speeds = sorted(np.unique(speed).tolist())
        if holdout_speed_min is None or holdout_speed_max is None:
            holdout_speed_min, holdout_speed_max = unique_speeds[-5], unique_speeds[-1]
        if holdout_speed_min > holdout_speed_max:
            raise ValueError("holdout_speed_min 不能大于 holdout_speed_max")

        test_mask = (speed >= holdout_speed_min) & (speed <= holdout_speed_max)
        if not np.any(test_mask):
            raise ValueError("测试速度区间没有样本")

        if val_speed_min is not None and val_speed_max is not None:
            val_mask = (speed >= val_speed_min) & (speed <= val_speed_max)
            if np.any(val_mask & test_mask):
                raise ValueError("验证速度区间不能与测试速度区间重叠")
            train_mask = ~(test_mask | val_mask)
            idx_train, idx_val, idx_test = make_split_from_masks(train_mask, val_mask, test_mask)
        else:
            remaining_idx = np.where(~test_mask)[0]
            idx_train_sub, idx_val_sub, _ = get_train_val_test_indices(
                X=X[remaining_idx],
                y=y[remaining_idx],
                test_size=0.0,
                val_size=val_size,
                random_state=random_state,
                use_stratify=True,
                stratify_view=stratify_view,
            )
            idx_train = remaining_idx[idx_train_sub]
            idx_val = remaining_idx[idx_val_sub]
            idx_test = np.where(test_mask)[0]
            idx_train, idx_val, idx_test = validate_predefined_split_indices(n_samples, idx_train, idx_val, idx_test)
        return {
            "protocol": protocol,
            "description": f"留一速度段外推：test 速度∈[{holdout_speed_min}, {holdout_speed_max}]",
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
            "holdout_speed_min": float(holdout_speed_min),
            "holdout_speed_max": float(holdout_speed_max),
            "val_speed_min": None if val_speed_min is None else float(val_speed_min),
            "val_speed_max": None if val_speed_max is None else float(val_speed_max),
            **get_stratify_metadata(stratify_view),
        }

    raise ValueError(f"不支持的 protocol：{protocol}")
