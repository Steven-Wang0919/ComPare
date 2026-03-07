# -*- coding: utf-8 -*-
"""
compare_all.py

综合对比六个模型：
- 正向：MLP / GRNN / KAN
- 反向：inverse_MLP / inverse_GRNN / inverse_KAN

职责：
1. 调用各自的 train_and_eval_* 函数
2. 保存正向指标表、正向预测表
3. 保存反向指标表（主结果 + 全测试集 + 主结果占比）
4. 保存反向全测试集预测表
5. 保存反向主结果子集预测表
6. 对同一任务内部不同模型的测试集真值进行强校验
7. 不允许通过静默截断来“对齐”结果

说明：
- 正向任务误差指标统一使用平均相对误差（ARE）
- 反向任务：
    主结果 = 策略一致子集
    补充结果 = 全测试集
- 如果三种反向模型的主结果子集不一致，将直接报错
"""

import os
import numpy as np
import pandas as pd

from train_mlp import train_and_eval_mlp
from train_grnn import train_and_eval_grnn
from train_kan import train_and_eval_kan

from inverse_mlp import train_and_eval_inverse_mlp
from inverse_grnn import train_and_eval_inverse_grnn
from inverse_kan import train_and_eval_inverse_kan_v2


# =========================
# 通用工具
# =========================
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
    """校验两个数组是否逐元素一致。"""
    if not np.allclose(arr1, arr2, atol=atol, rtol=rtol, equal_nan=False):
        diff_idx = np.where(
            ~np.isclose(arr1, arr2, atol=atol, rtol=rtol, equal_nan=False)
        )[0]
        first_idx = int(diff_idx[0])
        raise ValueError(
            f"{name1} 与 {name2} 的内容不一致，无法进行公平对比。\n"
            f"首个不一致位置: index={first_idx}, "
            f"{name1}={arr1[first_idx]}, {name2}={arr2[first_idx]}。\n"
            f"请检查测试集顺序、样本筛选或数据预处理是否一致。"
        )


def _validate_same_mask(mask1, mask2, name1, name2):
    """校验两个布尔掩码是否一致。"""
    m1 = np.asarray(mask1).astype(bool).reshape(-1)
    m2 = np.asarray(mask2).astype(bool).reshape(-1)

    _validate_same_length(m1, m2, name1, name2)

    if not np.array_equal(m1, m2):
        diff_idx = np.where(m1 != m2)[0]
        first_idx = int(diff_idx[0])
        raise ValueError(
            f"{name1} 与 {name2} 不一致，无法保证主结果子集可公平比较。\n"
            f"首个不一致位置: index={first_idx}, {name1}={m1[first_idx]}, {name2}={m2[first_idx]}。\n"
            f"请检查策略开度规则、测试集顺序或筛样逻辑是否一致。"
        )


def _fmt_float(x, ndigits=6):
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NaN"
        return f"{float(x):.{ndigits}g}"
    return str(x)


# =========================
# 正向任务汇总
# =========================
def run_forward_compare(output_dir):
    print("\n" + "=" * 72)
    print("开始正向模型对比：MLP / GRNN / KAN")
    print("=" * 72)

    mlp_res = train_and_eval_mlp()
    grnn_res = train_and_eval_grnn()
    kan_res = train_and_eval_kan()

    # ---------- 指标表 ----------
    metrics = [
        {
            "Task": "forward",
            "Model": "MLP",
            "R2": mlp_res["r2"],
            "ARE(%)": mlp_res["are"],
            "Hyperparams": (
                f"hidden={mlp_res.get('best_hidden')}, "
                f"alpha={mlp_res.get('best_alpha')}"
            ),
        },
        {
            "Task": "forward",
            "Model": "GRNN",
            "R2": grnn_res["r2"],
            "ARE(%)": grnn_res["are"],
            "Hyperparams": f"sigma={_fmt_float(grnn_res.get('best_sigma'))}",
        },
        {
            "Task": "forward",
            "Model": "KAN",
            "R2": kan_res["r2"],
            "ARE(%)": kan_res["are"],
            "Hyperparams": (
                f"hidden={kan_res.get('best_hidden_dim')}, "
                f"lr={_fmt_float(kan_res.get('best_lr'))}, "
                f"wd={_fmt_float(kan_res.get('best_weight_decay'))}"
            ),
        },
    ]

    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "forward_model_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n正向模型指标表已保存：{metrics_path}")

    # ---------- 强校验 ----------
    y_true_mlp = _to_1d_array(mlp_res["y_true"], "MLP y_true")
    y_true_grnn = _to_1d_array(grnn_res["y_true"], "GRNN y_true")
    y_true_kan = _to_1d_array(kan_res["y_true"], "KAN y_true")

    y_pred_mlp = _to_1d_array(mlp_res["y_pred"], "MLP y_pred")
    y_pred_grnn = _to_1d_array(grnn_res["y_pred"], "GRNN y_pred")
    y_pred_kan = _to_1d_array(kan_res["y_pred"], "KAN y_pred")

    _validate_same_length(y_true_mlp, y_pred_mlp, "MLP y_true", "MLP y_pred")
    _validate_same_length(y_true_grnn, y_pred_grnn, "GRNN y_true", "GRNN y_pred")
    _validate_same_length(y_true_kan, y_pred_kan, "KAN y_true", "KAN y_pred")

    _validate_same_length(y_true_mlp, y_true_grnn, "MLP y_true", "GRNN y_true")
    _validate_same_length(y_true_mlp, y_true_kan, "MLP y_true", "KAN y_true")

    _validate_same_values(y_true_mlp, y_true_grnn, "MLP y_true", "GRNN y_true")
    _validate_same_values(y_true_mlp, y_true_kan, "MLP y_true", "KAN y_true")

    # ---------- 预测表 ----------
    df_pred = pd.DataFrame({
        "true": y_true_mlp,
        "MLP_pred": y_pred_mlp,
        "GRNN_pred": y_pred_grnn,
        "KAN_pred": y_pred_kan,
    })

    pred_path = os.path.join(output_dir, "forward_model_predictions.csv")
    df_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"正向预测对比表已保存：{pred_path}")

    return {
        "metrics_df": df_metrics,
        "pred_df": df_pred,
    }


# =========================
# 反向任务汇总
# =========================
def run_inverse_compare(output_dir):
    print("\n" + "=" * 72)
    print("开始反向模型对比：inverse_MLP / inverse_GRNN / inverse_KAN")
    print("=" * 72)

    mlp_res = train_and_eval_inverse_mlp()
    grnn_res = train_and_eval_inverse_grnn()
    kan_res = train_and_eval_inverse_kan_v2()

    # ---------- 全测试集强校验 ----------
    y_true_all_mlp = _to_1d_array(mlp_res["y_true_all"], "inverse_MLP y_true_all")
    y_true_all_grnn = _to_1d_array(grnn_res["y_true_all"], "inverse_GRNN y_true_all")
    y_true_all_kan = _to_1d_array(kan_res["y_true_all"], "inverse_KAN y_true_all")

    y_pred_all_mlp = _to_1d_array(mlp_res["y_pred_all"], "inverse_MLP y_pred_all")
    y_pred_all_grnn = _to_1d_array(grnn_res["y_pred_all"], "inverse_GRNN y_pred_all")
    y_pred_all_kan = _to_1d_array(kan_res["y_pred_all"], "inverse_KAN y_pred_all")

    opening_all_mlp = _to_1d_array(mlp_res["opening_all"], "inverse_MLP opening_all")
    opening_all_grnn = _to_1d_array(grnn_res["opening_all"], "inverse_GRNN opening_all")
    opening_all_kan = _to_1d_array(kan_res["opening_all"], "inverse_KAN opening_all")

    mass_all_mlp = _to_1d_array(mlp_res["mass_all"], "inverse_MLP mass_all")
    mass_all_grnn = _to_1d_array(grnn_res["mass_all"], "inverse_GRNN mass_all")
    mass_all_kan = _to_1d_array(kan_res["mass_all"], "inverse_KAN mass_all")

    strat_open_all_mlp = _to_1d_array(mlp_res["strategy_opening_all"], "inverse_MLP strategy_opening_all")
    strat_open_all_grnn = _to_1d_array(grnn_res["strategy_opening_all"], "inverse_GRNN strategy_opening_all")
    strat_open_all_kan = _to_1d_array(kan_res["strategy_opening_all"], "inverse_KAN strategy_opening_all")

    policy_mask_mlp = np.asarray(mlp_res["policy_mask"]).astype(bool).reshape(-1)
    policy_mask_grnn = np.asarray(grnn_res["policy_mask"]).astype(bool).reshape(-1)
    policy_mask_kan = np.asarray(kan_res["policy_mask"]).astype(bool).reshape(-1)

    # 各模型内部长度匹配
    _validate_same_length(y_true_all_mlp, y_pred_all_mlp, "inverse_MLP y_true_all", "inverse_MLP y_pred_all")
    _validate_same_length(y_true_all_grnn, y_pred_all_grnn, "inverse_GRNN y_true_all", "inverse_GRNN y_pred_all")
    _validate_same_length(y_true_all_kan, y_pred_all_kan, "inverse_KAN y_true_all", "inverse_KAN y_pred_all")

    # 模型间全测试集一致
    _validate_same_length(y_true_all_mlp, y_true_all_grnn, "inverse_MLP y_true_all", "inverse_GRNN y_true_all")
    _validate_same_length(y_true_all_mlp, y_true_all_kan, "inverse_MLP y_true_all", "inverse_KAN y_true_all")

    _validate_same_values(y_true_all_mlp, y_true_all_grnn, "inverse_MLP y_true_all", "inverse_GRNN y_true_all")
    _validate_same_values(y_true_all_mlp, y_true_all_kan, "inverse_MLP y_true_all", "inverse_KAN y_true_all")

    _validate_same_values(mass_all_mlp, mass_all_grnn, "inverse_MLP mass_all", "inverse_GRNN mass_all")
    _validate_same_values(mass_all_mlp, mass_all_kan, "inverse_MLP mass_all", "inverse_KAN mass_all")

    _validate_same_values(opening_all_mlp, opening_all_grnn, "inverse_MLP opening_all", "inverse_GRNN opening_all")
    _validate_same_values(opening_all_mlp, opening_all_kan, "inverse_MLP opening_all", "inverse_KAN opening_all")

    _validate_same_values(
        strat_open_all_mlp, strat_open_all_grnn,
        "inverse_MLP strategy_opening_all", "inverse_GRNN strategy_opening_all"
    )
    _validate_same_values(
        strat_open_all_mlp, strat_open_all_kan,
        "inverse_MLP strategy_opening_all", "inverse_KAN strategy_opening_all"
    )

    # 主结果子集掩码也必须一致
    _validate_same_mask(policy_mask_mlp, policy_mask_grnn, "inverse_MLP policy_mask", "inverse_GRNN policy_mask")
    _validate_same_mask(policy_mask_mlp, policy_mask_kan, "inverse_MLP policy_mask", "inverse_KAN policy_mask")

    # ---------- 指标表 ----------
    metrics = [
        {
            "Task": "inverse",
            "Model": "inverse_MLP",
            "R2_main": mlp_res.get("r2_main"),
            "ARE_main(%)": mlp_res.get("are_main"),
            "n_main": mlp_res.get("n_main"),
            "main_ratio": mlp_res.get("main_ratio"),
            "R2_all": mlp_res.get("r2_all"),
            "ARE_all(%)": mlp_res.get("are_all"),
            "n_all": mlp_res.get("n_all"),
            "OpeningDist_all": str(mlp_res.get("opening_dist_all")),
            "OpeningDist_main": str(mlp_res.get("opening_dist_main")),
            "Hyperparams": (
                f"hidden={mlp_res.get('best_hidden')}, "
                f"alpha={mlp_res.get('best_alpha')}"
            ),
        },
        {
            "Task": "inverse",
            "Model": "inverse_GRNN",
            "R2_main": grnn_res.get("r2_main"),
            "ARE_main(%)": grnn_res.get("are_main"),
            "n_main": grnn_res.get("n_main"),
            "main_ratio": grnn_res.get("main_ratio"),
            "R2_all": grnn_res.get("r2_all"),
            "ARE_all(%)": grnn_res.get("are_all"),
            "n_all": grnn_res.get("n_all"),
            "OpeningDist_all": str(grnn_res.get("opening_dist_all")),
            "OpeningDist_main": str(grnn_res.get("opening_dist_main")),
            "Hyperparams": f"sigma={_fmt_float(grnn_res.get('best_sigma'))}",
        },
        {
            "Task": "inverse",
            "Model": "inverse_KAN",
            "R2_main": kan_res.get("r2_main"),
            "ARE_main(%)": kan_res.get("are_main"),
            "n_main": kan_res.get("n_main"),
            "main_ratio": kan_res.get("main_ratio"),
            "R2_all": kan_res.get("r2_all"),
            "ARE_all(%)": kan_res.get("are_all"),
            "n_all": kan_res.get("n_all"),
            "OpeningDist_all": str(kan_res.get("opening_dist_all")),
            "OpeningDist_main": str(kan_res.get("opening_dist_main")),
            "Hyperparams": (
                f"hidden={kan_res.get('best_hidden_dim')}, "
                f"lr={_fmt_float(kan_res.get('best_lr'))}, "
                f"wd={_fmt_float(kan_res.get('best_weight_decay'))}"
            ),
        },
    ]

    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "inverse_model_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n反向模型指标表已保存：{metrics_path}")

    # ---------- 全测试集预测表 ----------
    df_all_pred = pd.DataFrame({
        "target_mass": mass_all_mlp,
        "actual_opening": opening_all_mlp,
        "strategy_opening": strat_open_all_mlp,
        "policy_match": policy_mask_mlp.astype(int),
        "true_speed": y_true_all_mlp,
        "inverse_MLP_pred": y_pred_all_mlp,
        "inverse_GRNN_pred": y_pred_all_grnn,
        "inverse_KAN_pred": y_pred_all_kan,
    })

    all_pred_path = os.path.join(output_dir, "inverse_model_predictions_all.csv")
    df_all_pred.to_csv(all_pred_path, index=False, encoding="utf-8-sig")
    print(f"反向全测试集预测表已保存：{all_pred_path}")

    # ---------- 主结果子集预测表 ----------
    df_main_pred = df_all_pred[df_all_pred["policy_match"] == 1].copy()
    main_pred_path = os.path.join(output_dir, "inverse_model_predictions_main.csv")
    df_main_pred.to_csv(main_pred_path, index=False, encoding="utf-8-sig")
    print(f"反向主结果子集预测表已保存：{main_pred_path}")

    return {
        "metrics_df": df_metrics,
        "all_pred_df": df_all_pred,
        "main_pred_df": df_main_pred,
    }


# =========================
# 总入口
# =========================
def main():
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)

    forward_res = run_forward_compare(output_dir)
    inverse_res = run_inverse_compare(output_dir)

    # 额外输出一个总览表，便于论文表格整理
    overview_rows = []

    for _, row in forward_res["metrics_df"].iterrows():
        overview_rows.append({
            "Task": row["Task"],
            "Model": row["Model"],
            "Primary_R2": row["R2"],
            "Primary_ARE(%)": row["ARE(%)"],
            "n_main": np.nan,
            "main_ratio": np.nan,
            "Supplement_R2": np.nan,
            "Supplement_ARE(%)": np.nan,
        })

    for _, row in inverse_res["metrics_df"].iterrows():
        overview_rows.append({
            "Task": row["Task"],
            "Model": row["Model"],
            "Primary_R2": row["R2_main"],
            "Primary_ARE(%)": row["ARE_main(%)"],
            "n_main": row["n_main"],
            "main_ratio": row["main_ratio"],
            "Supplement_R2": row["R2_all"],
            "Supplement_ARE(%)": row["ARE_all(%)"],
        })

    df_overview = pd.DataFrame(overview_rows)
    overview_path = os.path.join(output_dir, "all_model_metrics_overview.csv")
    df_overview.to_csv(overview_path, index=False, encoding="utf-8-sig")
    print(f"\n总览指标表已保存：{overview_path}")

    print("\n" + "=" * 72)
    print("全部对比完成。")
    print("已输出：")
    print("  - forward_model_metrics.csv")
    print("  - forward_model_predictions.csv")
    print("  - inverse_model_metrics.csv")
    print("  - inverse_model_predictions_all.csv")
    print("  - inverse_model_predictions_main.csv")
    print("  - all_model_metrics_overview.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()