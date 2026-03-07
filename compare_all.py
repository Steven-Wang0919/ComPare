# -*- coding: utf-8 -*-
"""
综合对比三个模型（MLP / GRNN / KAN）

职责：
- 调用各自的 train_and_eval_* 函数
- 将指标表保存到： ./output_data/model_metrics.csv
- 将统一的真值 & 各模型预测保存到： ./output_data/model_predictions.csv

不画任何图，画图交给 plot_figures.py

说明：
- 三个模型的测试集样本必须完全一致，否则立即报错
- 不允许通过静默截断来“对齐”结果
- 误差指标统一使用平均相对误差（ARE）
"""

import os
import numpy as np
import pandas as pd

from train_mlp import train_and_eval_mlp
from train_grnn import train_and_eval_grnn
from train_kan import train_and_eval_kan


def _to_1d_array(arr, name):
    """将输入安全转换为一维 numpy 数组。"""
    out = np.asarray(arr).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} 为空，无法进行模型对比。")
    return out


def _validate_same_length(arr1, arr2, name1, name2):
    """校验两个数组长度是否一致。"""
    if len(arr1) != len(arr2):
        raise ValueError(
            f"{name1} 与 {name2} 的长度不一致："
            f"{name1}={len(arr1)}, {name2}={len(arr2)}。"
            f"请检查数据划分、过滤逻辑或推理流程是否一致。"
        )


def _validate_same_values(arr1, arr2, name1, name2, atol=1e-8, rtol=1e-6):
    """校验两个真值数组是否逐元素一致。"""
    if not np.allclose(arr1, arr2, atol=atol, rtol=rtol, equal_nan=False):
        diff_idx = np.where(
            ~np.isclose(arr1, arr2, atol=atol, rtol=rtol, equal_nan=False)
        )[0]
        first_idx = int(diff_idx[0])
        raise ValueError(
            f"{name1} 与 {name2} 的真值内容不一致，无法进行公平对比。\n"
            f"首个不一致位置: index={first_idx}, "
            f"{name1}={arr1[first_idx]}, {name2}={arr2[first_idx]}。\n"
            f"请检查测试集顺序、样本筛选或数据预处理是否一致。"
        )


def main():
    # 1. 创建输出文件夹
    data_dir = "output_data"
    os.makedirs(data_dir, exist_ok=True)

    # 2. 依次训练并评估三个模型
    mlp_res = train_and_eval_mlp()
    grnn_res = train_and_eval_grnn()
    kan_res = train_and_eval_kan()

    # 3. 构建指标表
    metrics = [
        {
            "Model": "MLP",
            "R2": mlp_res["r2"],
            "ARE(%)": mlp_res["are"],
            "Hyperparams": f"hidden={mlp_res.get('best_hidden')}, "
                           f"alpha={mlp_res.get('best_alpha')}",
        },
        {
            "Model": "GRNN",
            "R2": grnn_res["r2"],
            "ARE(%)": grnn_res["are"],
            "Hyperparams": f"sigma={grnn_res.get('best_sigma')}",
        },
        {
            "Model": "KAN",
            "R2": kan_res["r2"],
            "ARE(%)": kan_res["are"],
            "Hyperparams": f"hidden={kan_res.get('best_hidden_dim')}, "
                           f"lr={kan_res.get('best_lr')}, "
                           f"wd={kan_res.get('best_weight_decay')}",
        },
    ]

    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(data_dir, "model_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n模型指标表已保存：{metrics_path}")

    # 4. 强校验三个模型的真值与预测
    y_true_mlp = _to_1d_array(mlp_res["y_true"], "MLP y_true")
    y_true_grnn = _to_1d_array(grnn_res["y_true"], "GRNN y_true")
    y_true_kan = _to_1d_array(kan_res["y_true"], "KAN y_true")

    y_pred_mlp = _to_1d_array(mlp_res["y_pred"], "MLP y_pred")
    y_pred_grnn = _to_1d_array(grnn_res["y_pred"], "GRNN y_pred")
    y_pred_kan = _to_1d_array(kan_res["y_pred"], "KAN y_pred")

    # 先检查各模型内部 true/pred 长度是否匹配
    _validate_same_length(y_true_mlp, y_pred_mlp, "MLP y_true", "MLP y_pred")
    _validate_same_length(y_true_grnn, y_pred_grnn, "GRNN y_true", "GRNN y_pred")
    _validate_same_length(y_true_kan, y_pred_kan, "KAN y_true", "KAN y_pred")

    # 再检查不同模型之间的测试集长度和内容是否一致
    _validate_same_length(y_true_mlp, y_true_grnn, "MLP y_true", "GRNN y_true")
    _validate_same_length(y_true_mlp, y_true_kan, "MLP y_true", "KAN y_true")

    _validate_same_values(y_true_mlp, y_true_grnn, "MLP y_true", "GRNN y_true")
    _validate_same_values(y_true_mlp, y_true_kan, "MLP y_true", "KAN y_true")

    # 通过校验后，使用统一真值表
    y_true = y_true_mlp
    mlp_pred = y_pred_mlp
    grnn_pred = y_pred_grnn
    kan_pred = y_pred_kan

    df_pred = pd.DataFrame({
        "true": y_true,
        "MLP_pred": mlp_pred,
        "GRNN_pred": grnn_pred,
        "KAN_pred": kan_pred,
    })

    pred_path = os.path.join(data_dir, "model_predictions.csv")
    df_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"预测对比表已保存：{pred_path}")

    print("\n对比完成，可运行 plot_figures.py 生成图像。")


if __name__ == "__main__":
    main()