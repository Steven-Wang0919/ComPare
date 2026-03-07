# -*- coding: utf-8 -*-
"""
公共工具：
- 数据加载
- train/val/test 划分
- 指标函数
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def average_relative_error(y_true, y_pred, eps=1e-8):
    """
    平均相对误差（ARE, 百分比）

    定义：
        RE_i  = |(y_i - y_pred_i) / y_i| * 100%
        ARE   = mean(RE_i)

    说明：
    - 与施印炎等文中“相对误差 / 平均相对误差（ARE）”的写法对齐
    - 为避免分母过小导致数值不稳定，对 |y_true| < eps 的项使用 eps 替代
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
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
    X = df[["排肥口开度（mm）", "排肥轴转速（r/min）"]].values.astype(np.float32)
    y = df["实际排肥质量（g/min）"].values.astype(np.float32)
    return X, y


def get_train_val_test_indices(
    n_samples,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """
    统一 train/val/test 三分数据，保证三个模型使用完全相同的样本索引。

    默认约：train 70%, val 15%, test 15%
    """
    idx_all = np.arange(n_samples)

    # 先划出 test
    idx_train_val, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=random_state
    )

    # 再在剩余里划出 val，val_size 是“相对总体”的比例
    val_size_rel = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=val_size_rel, random_state=random_state
    )

    return idx_train, idx_val, idx_test