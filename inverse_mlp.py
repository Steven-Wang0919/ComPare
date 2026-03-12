# -*- coding: utf-8 -*-
"""
inverse_mlp.py

反向 MLP：
- 主结果 = 策略一致子集
- 补充结果 = 全测试集
- 默认仅返回结果，不再写仓库根目录
- 独立运行时输出到 runs/<timestamp>_inverse_mlp/
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from common_utils import load_data, get_train_val_test_indices, average_relative_error
from run_utils import append_manifest_outputs, create_run_dir, save_dataframe, write_manifest


THRESHOLD_LOW_MID = 2800.0
THRESHOLD_MID_HIGH = 4800.0
EPS = 1e-8


def select_optimal_opening(target_mass: float) -> float:
    if target_mass < THRESHOLD_LOW_MID:
        return 20.0
    elif target_mass < THRESHOLD_MID_HIGH:
        return 35.0
    else:
        return 50.0


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


def _count_openings(openings, opening_values=(20.0, 35.0, 50.0), atol=0.1):
    openings = np.asarray(openings, dtype=float)
    stats = {}
    for v in opening_values:
        stats[f"{int(v)}mm"] = int(np.isclose(openings, v, atol=atol).sum())
    stats["other"] = int(len(openings) - sum(stats[k] for k in ["20mm", "35mm", "50mm"]))
    return stats


def train_and_eval_inverse_mlp(
    data_path="data/dataset.xlsx",
    hidden_layer_candidates=None,
    alpha_candidates=None,
    max_iter=5000,
    random_state=0,
    save_outputs_dir=None,
):
    if hidden_layer_candidates is None:
        hidden_layer_candidates = [(10,), (20,), (50,), (20, 20)]
    if alpha_candidates is None:
        alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    print("\n=== 训练 反向 MLP（train-only normalization，转速优先口径） ===")

    X_raw, y_raw = load_data(data_path)
    idx_tr, idx_val, idx_te = get_train_val_test_indices(X=X_raw, y=y_raw)

    X_inv_all = np.stack([y_raw, X_raw[:, 0]], axis=1)
    y_inv_all = X_raw[:, 1]

    X_train_raw = X_inv_all[idx_tr]
    X_val_raw = X_inv_all[idx_val]
    X_test_raw = X_inv_all[idx_te]

    y_train_raw = y_inv_all[idx_tr]
    y_val_raw = y_inv_all[idx_val]

    X_min = X_train_raw.min(axis=0, keepdims=True)
    X_max = X_train_raw.max(axis=0, keepdims=True)
    y_min = float(y_train_raw.min())
    y_max = float(y_train_raw.max())

    def norm_x(x):
        return (x - X_min) / (X_max - X_min + EPS)

    def norm_y(v):
        return (v - y_min) / (y_max - y_min + EPS)

    def denorm_y(v):
        return v * (y_max - y_min + EPS) + y_min

    X_train = norm_x(X_train_raw)
    X_val = norm_x(X_val_raw)
    y_train_norm = norm_y(y_train_raw)

    best_r2_val = -np.inf
    best_hidden = None
    best_alpha = None

    for h in hidden_layer_candidates:
        for a in alpha_candidates:
            model = MLPRegressor(
                hidden_layer_sizes=h,
                alpha=a,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )
            model.fit(X_train, y_train_norm)

            y_val_pred_norm = model.predict(X_val)
            y_val_pred = denorm_y(y_val_pred_norm)
            r2_val = r2_score(y_val_raw, y_val_pred)

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_hidden = h
                best_alpha = a

    if best_hidden is None or best_alpha is None:
        raise RuntimeError("反向 MLP 超参数搜索失败：best_hidden / best_alpha 为空")

    print(
        f"反向 MLP 最优超参数：hidden_layer_sizes={best_hidden}, "
        f"alpha={best_alpha}, solver='lbfgs', val R²={best_r2_val:.6f}"
    )

    X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_val_raw = np.hstack([y_train_raw, y_val_raw])

    X_train_val = norm_x(X_train_val_raw)
    y_train_val_norm = norm_y(y_train_val_raw)

    mlp_final = MLPRegressor(
        hidden_layer_sizes=best_hidden,
        alpha=best_alpha,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp_final.fit(X_train_val, y_train_val_norm)

    test_mass = y_raw[idx_te]
    test_opening = X_raw[idx_te, 0]
    test_speed_true = X_raw[idx_te, 1]

    X_test_norm = norm_x(X_test_raw)
    pred_test_norm = mlp_final.predict(X_test_norm)
    pred_test_speed = denorm_y(pred_test_norm)

    r2_all = _safe_r2(test_speed_true, pred_test_speed)
    are_all = _safe_are(test_speed_true, pred_test_speed)
    n_all = int(len(test_speed_true))

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
        y_true_main = test_speed_true[policy_mask]
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

    print("\n===== 反向 MLP（转速优先口径）测试结果 =====")
    print(
        f"主结果: n={n_main}/{n_all}, R²={r2_main}, ARE={are_main}% | "
        f"全测试集: R²={r2_all}, ARE={are_all}%"
    )

    if save_outputs_dir is not None:
        df_all = pd.DataFrame({
            "target_mass_g_min": test_mass,
            "actual_opening_mm": test_opening,
            "strategy_opening_mm": strategy_opening_all,
            "true_speed_r_min": test_speed_true,
            "inverse_MLP_pred": pred_test_speed,
            "policy_match": policy_mask.astype(int),
        })
        save_dataframe(df_all, os.path.join(save_outputs_dir, "inverse_mlp_predictions_all.csv"))

        df_main = pd.DataFrame({
            "target_mass_g_min": mass_main,
            "actual_opening_mm": opening_main,
            "strategy_opening_mm": strategy_opening_main,
            "true_speed_r_min": y_true_main,
            "inverse_MLP_pred": y_pred_main,
        })
        save_dataframe(df_main, os.path.join(save_outputs_dir, "inverse_mlp_predictions_main.csv"))

    return {
        "r2_main": r2_main,
        "are_main": are_main,
        "n_main": n_main,
        "main_ratio": main_ratio,
        "r2_all": r2_all,
        "are_all": are_all,
        "n_all": n_all,
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,
        "y_true_all": np.asarray(test_speed_true),
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
    }


def main():
    run_dir = create_run_dir("inverse_mlp")

    write_manifest(
        run_dir,
        script_name="inverse_mlp.py",
        data_path="data/dataset.xlsx",
        seed=0,
        params={
            "hidden_layer_candidates": [(10,), (20,), (50,), (20, 20)],
            "alpha_candidates": [1e-6, 1e-5, 1e-4, 1e-3],
            "max_iter": 5000,
            "random_state": 0,
        },
    )

    train_and_eval_inverse_mlp(save_outputs_dir=run_dir)

    append_manifest_outputs(
        run_dir,
        [
            {"path": "inverse_mlp_predictions_all.csv"},
            {"path": "inverse_mlp_predictions_main.csv"},
        ],
    )

    print(f"\n本次运行目录：{run_dir}")


if __name__ == "__main__":
    main()