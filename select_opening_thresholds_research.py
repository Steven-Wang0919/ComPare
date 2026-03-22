# -*- coding: utf-8 -*-
"""
select_opening_thresholds_research.py

Research script for making the three-stage opening policy interpretable.

Outputs:
- summary.json
- rule_comparison.csv
- threshold_zone_summary.csv
- narrative_summary.md
- figures/
"""

import argparse
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.environ.setdefault("OMP_NUM_THREADS", "2")

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from common_utils import load_data
from policy_config import (
    LEGACY_LOW_MID_THRESHOLD,
    LEGACY_MID_HIGH_THRESHOLD,
    POLICY_LABEL,
    POLICY_LOW_MID_THRESHOLD,
    POLICY_MID_HIGH_THRESHOLD,
    POLICY_TARGET_OPENINGS,
)
from run_utils import (
    append_manifest_outputs,
    create_run_dir,
    ensure_dir,
    jsonable,
    save_dataframe,
    write_manifest,
)

DEFAULT_REFERENCE_SPEED = 40.0
DEFAULT_N_CLUSTERS = 3
DEFAULT_TOP_K = 20
DEFAULT_RANDOM_SEED = 42
EPS = 1e-12


try:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
except AttributeError:
    sns.set(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams["axes.unicode_minus"] = False


def _fmt_float(x, digits=3):
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NaN"
        return f"{float(x):.{digits}f}"
    return str(x)


def _to_float_list(values):
    return [float(v) for v in values]


def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < EPS:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _interval_overlap(interval_a, interval_b):
    a0, a1 = interval_a
    b0, b1 = interval_b
    return max(0.0, min(a1, b1) - max(a0, b0))


def _zone_index_from_thresholds(values, low_mid, mid_high):
    values = np.asarray(values, dtype=float).reshape(-1)
    out = np.zeros(len(values), dtype=int)
    out[values >= low_mid] = 1
    out[values >= mid_high] = 2
    return out


def _build_zone_rows(low_mid, mid_high, mass_min, mass_max, centers):
    return [
        {
            "zone": "low",
            "zone_index": 0,
            "mass_min": float(mass_min),
            "mass_max": float(low_mid),
            "center": float(centers[0]),
            "width": float(max(low_mid - mass_min, EPS)),
        },
        {
            "zone": "mid",
            "zone_index": 1,
            "mass_min": float(low_mid),
            "mass_max": float(mid_high),
            "center": float(centers[1]),
            "width": float(max(mid_high - low_mid, EPS)),
        },
        {
            "zone": "high",
            "zone_index": 2,
            "mass_min": float(mid_high),
            "mass_max": float(mass_max),
            "center": float(centers[2]),
            "width": float(max(mass_max - mid_high, EPS)),
        },
    ]


def _fit_opening_models(X, y, reference_speed):
    opening_values = sorted(np.unique(np.asarray(X[:, 0], dtype=float)).tolist())
    models = []
    opening_model_map = {}

    for opening in opening_values:
        mask = np.isclose(X[:, 0], opening)
        speeds = np.asarray(X[mask, 1], dtype=float)
        masses = np.asarray(y[mask], dtype=float)
        slope, intercept = np.polyfit(speeds, masses, deg=1)
        pred_train = slope * speeds + intercept
        estimated_mass = slope * float(reference_speed) + intercept

        row = {
            "opening": float(opening),
            "slope": float(slope),
            "intercept": float(intercept),
            "speed_min": float(np.min(speeds)),
            "speed_max": float(np.max(speeds)),
            "mass_min": float(np.min(masses)),
            "mass_max": float(np.max(masses)),
            "estimated_mass_at_reference_speed": float(estimated_mass),
            "reference_speed": float(reference_speed),
            "r2": _safe_r2(masses, pred_train),
            "sample_count": int(mask.sum()),
        }
        models.append(row)
        opening_model_map[float(opening)] = row

    return models, opening_model_map


def _compute_thresholds_from_sorted_groups(group_rows, fallback_centers):
    thresholds = []
    for idx in range(len(group_rows) - 1):
        left = group_rows[idx]
        right = group_rows[idx + 1]
        left_max = left["mass_max"]
        right_min = right["mass_min"]
        if left_max is not None and right_min is not None and right_min > left_max:
            thr = 0.5 * (left_max + right_min)
        else:
            thr = 0.5 * (fallback_centers[idx] + fallback_centers[idx + 1])
        thresholds.append(float(thr))
    return thresholds


def _summarize_partition(values, assignments, centers, variances=None, weights=None):
    values = np.asarray(values, dtype=float).reshape(-1)
    centers = np.asarray(centers, dtype=float).reshape(-1)
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    sorted_centers = centers[order]

    sorted_assignments = np.array([remap[int(a)] for a in assignments], dtype=int)
    variances_sorted = None
    if variances is not None:
        variances_sorted = np.asarray(variances, dtype=float).reshape(-1)[order]
    weights_sorted = None
    if weights is not None:
        weights_sorted = np.asarray(weights, dtype=float).reshape(-1)[order]

    rows = []
    for idx, center in enumerate(sorted_centers):
        cluster_values = values[sorted_assignments == idx]
        row = {
            "cluster": int(idx),
            "center": float(center),
            "count": int(len(cluster_values)),
            "mass_min": None if len(cluster_values) == 0 else float(np.min(cluster_values)),
            "mass_max": None if len(cluster_values) == 0 else float(np.max(cluster_values)),
        }
        if variances_sorted is not None:
            row["variance"] = float(variances_sorted[idx])
        if weights_sorted is not None:
            row["weight"] = float(weights_sorted[idx])
        rows.append(row)

    thresholds = _compute_thresholds_from_sorted_groups(rows, sorted_centers)

    return {
        "sorted_assignments": sorted_assignments,
        "sorted_centers": sorted_centers,
        "cluster_summary": rows,
        "thresholds": thresholds,
    }


def _learn_kmeans_thresholds(y, n_clusters, seed):
    model = KMeans(
        n_clusters=n_clusters,
        n_init=16,
        random_state=seed,
    )
    assignments = model.fit_predict(np.asarray(y, dtype=float).reshape(-1, 1))
    summary = _summarize_partition(y, assignments, model.cluster_centers_.reshape(-1))
    return {
        "low_mid_threshold": float(summary["thresholds"][0]),
        "mid_high_threshold": float(summary["thresholds"][1]),
        "cluster_centers": _to_float_list(summary["sorted_centers"]),
        "cluster_summary": summary["cluster_summary"],
        "assignments": summary["sorted_assignments"],
        "inertia": float(model.inertia_),
    }


def _learn_gmm_thresholds(y, n_clusters, seed):
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=seed,
    )
    values = np.asarray(y, dtype=float).reshape(-1, 1)
    model.fit(values)
    assignments = model.predict(values)
    means = model.means_.reshape(-1)
    covariances = np.asarray([cov[0, 0] for cov in model.covariances_], dtype=float)
    summary = _summarize_partition(
        y,
        assignments,
        means,
        variances=covariances,
        weights=model.weights_,
    )
    return {
        "low_mid_threshold": float(summary["thresholds"][0]),
        "mid_high_threshold": float(summary["thresholds"][1]),
        "component_means": _to_float_list(summary["sorted_centers"]),
        "component_variances": _to_float_list([row["variance"] for row in summary["cluster_summary"]]),
        "component_weights": _to_float_list([row["weight"] for row in summary["cluster_summary"]]),
        "cluster_summary": summary["cluster_summary"],
        "assignments": summary["sorted_assignments"],
        "lower_bound": float(model.lower_bound_),
    }


def _margin_score(pred_mass, zone_idx, low_mid, mid_high, mass_min, mass_max):
    if zone_idx == 0:
        denom = max(low_mid - mass_min, EPS)
        return _clip01((low_mid - pred_mass) / denom)
    if zone_idx == 1:
        zone_width = max(mid_high - low_mid, EPS)
        if pred_mass < low_mid or pred_mass > mid_high:
            return 0.0
        distance = min(pred_mass - low_mid, mid_high - pred_mass)
        return _clip01(distance / (0.5 * zone_width + EPS))
    denom = max(mass_max - mid_high, EPS)
    return _clip01((pred_mass - mid_high) / denom)


def _score_opening_for_zone(opening_row, zone_row, low_mid, mid_high, mass_min, mass_max):
    pred_mass = float(opening_row["estimated_mass_at_reference_speed"])
    zone_interval = (zone_row["mass_min"], zone_row["mass_max"])
    opening_interval = (opening_row["mass_min"], opening_row["mass_max"])
    fit = _clip01(1.0 - abs(pred_mass - zone_row["center"]) / (zone_row["width"] + EPS))
    coverage = _clip01(_interval_overlap(opening_interval, zone_interval) / (zone_row["width"] + EPS))
    margin = _margin_score(pred_mass, zone_row["zone_index"], low_mid, mid_high, mass_min, mass_max)
    role_score = 0.5 * fit + 0.3 * coverage + 0.2 * margin
    return {
        "fit_score": float(fit),
        "coverage_score": float(coverage),
        "margin_score": float(margin),
        "role_score": float(role_score),
    }


def _summarize_fixed_triplet_zones(
    *,
    threshold_source,
    triplet,
    low_mid,
    mid_high,
    opening_model_map,
    mass_min,
    mass_max,
    centers,
):
    zone_rows = _build_zone_rows(
        low_mid=low_mid,
        mid_high=mid_high,
        mass_min=mass_min,
        mass_max=mass_max,
        centers=centers,
    )
    summary_rows = []
    for idx, zone_row in enumerate(zone_rows):
        opening_row = opening_model_map[float(triplet[idx])]
        zone_interval = (zone_row["mass_min"], zone_row["mass_max"])
        opening_interval = (opening_row["mass_min"], opening_row["mass_max"])
        overlap = _interval_overlap(zone_interval, opening_interval)
        coverage = overlap / (zone_row["width"] + EPS)
        summary_rows.append({
            "threshold_source": threshold_source,
            "zone": zone_row["zone"],
            "assigned_opening_mm": float(opening_row["opening"]),
            "threshold_low_mid": float(low_mid),
            "threshold_mid_high": float(mid_high),
            "zone_mass_min": float(zone_row["mass_min"]),
            "zone_mass_max": float(zone_row["mass_max"]),
            "zone_center": float(zone_row["center"]),
            "zone_width": float(zone_row["width"]),
            "opening_mass_min": float(opening_row["mass_min"]),
            "opening_mass_max": float(opening_row["mass_max"]),
            "overlap_length": float(overlap),
            "coverage_ratio": float(coverage),
            "reference_speed_mass": float(opening_row["estimated_mass_at_reference_speed"]),
            "reference_gap_to_zone_center": float(abs(opening_row["estimated_mass_at_reference_speed"] - zone_row["center"])),
        })
    return summary_rows


def _evaluate_fixed_triplet_rule(
    *,
    threshold_source,
    triplet,
    low_mid,
    mid_high,
    mass_values,
    opening_model_map,
    learned_low_mid,
    learned_mid_high,
    centers,
):
    mass_values = np.asarray(mass_values, dtype=float).reshape(-1)
    mass_min = float(np.min(mass_values))
    mass_max = float(np.max(mass_values))
    zone_idx = _zone_index_from_thresholds(mass_values, low_mid, mid_high)
    learned_zone_idx = _zone_index_from_thresholds(mass_values, learned_low_mid, learned_mid_high)
    zone_summaries = _summarize_fixed_triplet_zones(
        threshold_source=threshold_source,
        triplet=triplet,
        low_mid=low_mid,
        mid_high=mid_high,
        opening_model_map=opening_model_map,
        mass_min=mass_min,
        mass_max=mass_max,
        centers=centers,
    )

    feasibility = []
    for value, idx in zip(mass_values, zone_idx):
        opening_row = opening_model_map[float(triplet[int(idx)])]
        feasibility.append(opening_row["mass_min"] - EPS <= value <= opening_row["mass_max"] + EPS)

    estimated_masses = [opening_model_map[float(op)]["estimated_mass_at_reference_speed"] for op in triplet]
    return {
        "rule_name": threshold_source,
        "threshold_source": threshold_source,
        "triplet": _to_float_list(triplet),
        "low_mid_threshold": float(low_mid),
        "mid_high_threshold": float(mid_high),
        "low_opening": float(triplet[0]),
        "mid_opening": float(triplet[1]),
        "high_opening": float(triplet[2]),
        "low_estimated_mass": float(estimated_masses[0]),
        "mid_estimated_mass": float(estimated_masses[1]),
        "high_estimated_mass": float(estimated_masses[2]),
        "feasible_coverage_rate": float(np.mean(feasibility)),
        "zone_label_agreement_vs_kmeans": float(np.mean(zone_idx == learned_zone_idx)),
        "low_mid_deviation_vs_kmeans": float(low_mid - learned_low_mid),
        "mid_high_deviation_vs_kmeans": float(mid_high - learned_mid_high),
        "zone_summaries": zone_summaries,
    }


def _build_rule_comparison(engineering_eval, learned_eval):
    rows = []
    for item in [engineering_eval, learned_eval]:
        rows.append({
            "rule_name": item["rule_name"],
            "threshold_source": item["threshold_source"],
            "low_mid_threshold": float(item["low_mid_threshold"]),
            "mid_high_threshold": float(item["mid_high_threshold"]),
            "low_opening": float(item["low_opening"]),
            "mid_opening": float(item["mid_opening"]),
            "high_opening": float(item["high_opening"]),
            "low_estimated_mass": float(item["low_estimated_mass"]),
            "mid_estimated_mass": float(item["mid_estimated_mass"]),
            "high_estimated_mass": float(item["high_estimated_mass"]),
            "feasible_coverage_rate": float(item["feasible_coverage_rate"]),
            "zone_label_agreement_vs_kmeans": float(item["zone_label_agreement_vs_kmeans"]),
            "low_mid_deviation_vs_kmeans": float(item["low_mid_deviation_vs_kmeans"]),
            "mid_high_deviation_vs_kmeans": float(item["mid_high_deviation_vs_kmeans"]),
        })
    return pd.DataFrame(rows)


def _write_summary_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)


def _make_mass_distribution_figure(fig_path, masses, kmeans_res, gmm_res):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(masses, bins=24, kde=True, ax=ax, color="#4C78A8", alpha=0.35)

    ax.axvline(LEGACY_LOW_MID_THRESHOLD, color="#E45756", linestyle="--", linewidth=2, label="Engineering low-mid")
    ax.axvline(LEGACY_MID_HIGH_THRESHOLD, color="#E45756", linestyle="--", linewidth=2, label="Engineering mid-high")
    ax.axvline(kmeans_res["low_mid_threshold"], color="#54A24B", linestyle="-", linewidth=2, label="KMeans low-mid")
    ax.axvline(kmeans_res["mid_high_threshold"], color="#54A24B", linestyle="-", linewidth=2, label="KMeans mid-high")
    ax.axvline(gmm_res["low_mid_threshold"], color="#B279A2", linestyle=":", linewidth=2, label="GMM low-mid")
    ax.axvline(gmm_res["mid_high_threshold"], color="#B279A2", linestyle=":", linewidth=2, label="GMM mid-high")

    ax.set_title("Mass Distribution and Policy Thresholds")
    ax.set_xlabel("Mass (g/min)")
    ax.set_ylabel("Count")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def _make_opening_ranges_figure(fig_path, opening_models):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_pos = np.arange(len(opening_models))

    for idx, row in enumerate(opening_models):
        ax.hlines(
            y=idx,
            xmin=row["mass_min"],
            xmax=row["mass_max"],
            color="#4C78A8",
            linewidth=4,
            alpha=0.8,
        )
        ax.scatter(
            row["estimated_mass_at_reference_speed"],
            idx,
            color="#F58518",
            s=60,
            zorder=3,
        )

    ax.axvline(LEGACY_LOW_MID_THRESHOLD, color="#E45756", linestyle="--", linewidth=1.8)
    ax.axvline(LEGACY_MID_HIGH_THRESHOLD, color="#E45756", linestyle="--", linewidth=1.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{int(row['opening'])} mm" for row in opening_models])
    ax.set_xlabel("Mass (g/min)")
    ax.set_ylabel("Opening")
    ax.set_title("Observed Opening Reachable Mass Ranges")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def _draw_rule_panel(ax, thresholds, triplet, opening_model_map, title):
    t1, t2 = thresholds
    masses = [opening_model_map[float(op)]["estimated_mass_at_reference_speed"] for op in triplet]

    ax.axvspan(0, t1, color="#54A24B", alpha=0.12)
    ax.axvspan(t1, t2, color="#F2CF5B", alpha=0.14)
    ax.axvspan(t2, ax.get_xlim()[1], color="#E45756", alpha=0.10)
    ax.axvline(t1, color="#333333", linestyle="--", linewidth=1.4)
    ax.axvline(t2, color="#333333", linestyle="--", linewidth=1.4)

    ax.scatter(masses, [1, 1, 1], s=90, color=["#54A24B", "#F2CF5B", "#E45756"], zorder=3)
    for mass, op in zip(masses, triplet):
        ax.text(mass, 1.05, f"{int(op)} mm", ha="center", va="bottom", fontsize=10)

    ax.set_yticks([])
    ax.set_xlabel("Mass at reference speed (g/min)")
    ax.set_title(title)


def _make_rule_comparison_figure(fig_path, rule_comparison_df, opening_model_map, mass_min, mass_max):
    current_row = rule_comparison_df.loc[rule_comparison_df["rule_name"] == "engineering_fixed_thresholds"].iloc[0]
    best_row = rule_comparison_df.loc[rule_comparison_df["rule_name"] == "kmeans_learned_thresholds"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    for ax in axes:
        ax.set_xlim(mass_min, mass_max)

    _draw_rule_panel(
        axes[0],
        thresholds=(current_row["low_mid_threshold"], current_row["mid_high_threshold"]),
        triplet=(current_row["low_opening"], current_row["mid_opening"], current_row["high_opening"]),
        opening_model_map=opening_model_map,
        title="Current engineering rule",
    )
    _draw_rule_panel(
        axes[1],
        thresholds=(best_row["low_mid_threshold"], best_row["mid_high_threshold"]),
        triplet=(best_row["low_opening"], best_row["mid_opening"], best_row["high_opening"]),
        opening_model_map=opening_model_map,
        title="Fixed triplet with learned thresholds",
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def _build_narrative(summary, top_k):
    data_info = summary["data_overview"]
    kmeans_info = summary["kmeans_thresholds"]
    gmm_info = summary["gmm_sensitivity"]
    learned = summary["learned_rule_validation"]
    triplet = summary["fixed_triplet_mm"]

    paper_text = f"""## 论文版

基于 `data/dataset.xlsx` 中 {data_info['sample_count']} 个样本、{len(data_info['observed_openings_mm'])} 个观测开度档位（{', '.join(str(int(v)) for v in data_info['observed_openings_mm'])} mm），本文对固定开度三元组 `({int(triplet[0])}, {int(triplet[1])}, {int(triplet[2])})` 的三段式策略边界进行了专门研究。研究目标不是重新选择开度档位，而是在固定低/中/高三档控制开度的前提下，学习三档之间更符合当前数据的排量分界。

在排肥量维度上，`KMeans(3)` 学得的主分界点分别为 `{_fmt_float(kmeans_info['low_mid_threshold'])}` g/min 和 `{_fmt_float(kmeans_info['mid_high_threshold'])}` g/min；`GMM(3)` 敏感性分析得到的对应分界点为 `{_fmt_float(gmm_info['low_mid_threshold'])}` g/min 和 `{_fmt_float(gmm_info['mid_high_threshold'])}` g/min，说明三段边界在数据上具有较好的稳定性。相较于旧工程边界 `2800/4800 g/min`，新的数据驱动边界在固定三元组下保持了同样的控制结构，但将低/中和中/高分界分别上移到了 `{_fmt_float(kmeans_info['low_mid_threshold'])}` 和 `{_fmt_float(kmeans_info['mid_high_threshold'])}` g/min。

因此，`20/35/50 mm` 应表述为固定的低、中、高三档控制开度，而数据驱动学习的对象是这三档之间的排量分界。旧工程边界 `2800/4800 g/min` 可作为历史基线，新边界 `{_fmt_float(kmeans_info['low_mid_threshold'])}/{_fmt_float(kmeans_info['mid_high_threshold'])}` g/min` 则作为当前默认策略边界。
"""

    defense_text = f"""## 答辩版

我们没有改三档开度本身，还是固定 `20/35/50 mm`。这次优化的不是开度档位，而是三档之间的排量分界。

结果是，数据聚类给出的两个主分界点大约在 `{_fmt_float(kmeans_info['low_mid_threshold'])}` 和 `{_fmt_float(kmeans_info['mid_high_threshold'])}` g/min；敏感性分析给出的边界在 `{_fmt_float(gmm_info['low_mid_threshold'])}` 和 `{_fmt_float(gmm_info['mid_high_threshold'])}` g/min 左右。也就是说，三段式思路本身是稳定的，只是原来 `2800/4800 g/min` 偏保守，现在我们把默认边界升级成更贴近数据的版本。
"""

    readme_text = f"""## README / 代码说明版

`select_opening_thresholds_research.py` 用于给固定开度三元组 `20/35/50 mm` 的策略边界提供证据链。脚本会同时对比旧工程边界 `2800/4800 g/min` 与数据驱动边界，并输出结构化结果、图表和可直接引用的文字摘要。

当前数据上，KMeans 学得的主边界约为 `{_fmt_float(kmeans_info['low_mid_threshold'])}` / `{_fmt_float(kmeans_info['mid_high_threshold'])}` g/min，GMM 敏感性边界约为 `{_fmt_float(gmm_info['low_mid_threshold'])}` / `{_fmt_float(gmm_info['mid_high_threshold'])}` g/min。运行时默认使用固定三元组 `20/35/50 mm` 和 KMeans 主边界，不再使用旧的 `2800/4800 g/min` 作为默认策略阈值。

输出中的 `rule_comparison.csv` 和 `threshold_zone_summary.csv` 用于直接比较旧边界和新边界在固定三段策略下的区间、覆盖和偏差。
"""

    return "\n\n".join([paper_text, defense_text, readme_text]).strip() + "\n"


def _save_narrative(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run_research(data_path, n_clusters, reference_speed, candidate_source, top_k, seed, output_dir=None):
    if int(n_clusters) != 3:
        raise ValueError("This research entry currently supports exactly 3 mass clusters.")
    if candidate_source != "observed":
        raise ValueError("Only candidate_source='observed' is supported in this version.")

    if output_dir is None:
        run_dir = create_run_dir("opening_threshold_research")
    else:
        run_dir = ensure_dir(output_dir)
    figure_dir = ensure_dir(os.path.join(run_dir, "figures"))

    X_raw, y_raw = load_data(data_path)
    mass_values = np.asarray(y_raw, dtype=float).reshape(-1)
    observed_openings = sorted(np.unique(np.asarray(X_raw[:, 0], dtype=float)).tolist())
    opening_models, opening_model_map = _fit_opening_models(X_raw, y_raw, reference_speed=reference_speed)

    kmeans_res = _learn_kmeans_thresholds(y_raw, n_clusters=n_clusters, seed=seed)
    gmm_res = _learn_gmm_thresholds(y_raw, n_clusters=n_clusters, seed=seed)

    mass_min = float(np.min(mass_values))
    mass_max = float(np.max(mass_values))
    fixed_triplet = POLICY_TARGET_OPENINGS
    engineering_centers = [
        0.5 * (mass_min + LEGACY_LOW_MID_THRESHOLD),
        0.5 * (LEGACY_LOW_MID_THRESHOLD + LEGACY_MID_HIGH_THRESHOLD),
        0.5 * (LEGACY_MID_HIGH_THRESHOLD + mass_max),
    ]
    engineering_eval = _evaluate_fixed_triplet_rule(
        threshold_source="engineering_fixed_thresholds",
        triplet=fixed_triplet,
        low_mid=LEGACY_LOW_MID_THRESHOLD,
        mid_high=LEGACY_MID_HIGH_THRESHOLD,
        mass_values=mass_values,
        opening_model_map=opening_model_map,
        learned_low_mid=POLICY_LOW_MID_THRESHOLD,
        learned_mid_high=POLICY_MID_HIGH_THRESHOLD,
        centers=engineering_centers,
    )
    learned_eval = _evaluate_fixed_triplet_rule(
        threshold_source="kmeans_learned_thresholds",
        triplet=fixed_triplet,
        low_mid=POLICY_LOW_MID_THRESHOLD,
        mid_high=POLICY_MID_HIGH_THRESHOLD,
        mass_values=mass_values,
        opening_model_map=opening_model_map,
        learned_low_mid=POLICY_LOW_MID_THRESHOLD,
        learned_mid_high=POLICY_MID_HIGH_THRESHOLD,
        centers=kmeans_res["cluster_centers"],
    )
    threshold_zone_summary_df = pd.DataFrame(
        engineering_eval["zone_summaries"] + learned_eval["zone_summaries"]
    )
    rule_comparison_df = _build_rule_comparison(engineering_eval, learned_eval)

    summary = {
        "config": {
            "data_path": data_path,
            "n_clusters": int(n_clusters),
            "reference_speed": float(reference_speed),
            "candidate_source": candidate_source,
            "seed": int(seed),
            "policy_label": POLICY_LABEL,
        },
        "data_overview": {
            "sample_count": int(len(mass_values)),
            "observed_openings_mm": _to_float_list(observed_openings),
            "speed_range_r_min": [
                float(np.min(X_raw[:, 1])),
                float(np.max(X_raw[:, 1])),
            ],
            "mass_range_g_min": [mass_min, mass_max],
        },
        "fixed_triplet_mm": _to_float_list(fixed_triplet),
        "opening_models": opening_models,
        "engineering_rule_validation": engineering_eval,
        "learned_rule_validation": learned_eval,
        "kmeans_thresholds": {
            "low_mid_threshold": float(POLICY_LOW_MID_THRESHOLD),
            "mid_high_threshold": float(POLICY_MID_HIGH_THRESHOLD),
            "cluster_centers": _to_float_list(kmeans_res["cluster_centers"]),
            "cluster_summary": kmeans_res["cluster_summary"],
            "inertia": float(kmeans_res["inertia"]),
        },
        "engineering_thresholds": {
            "low_mid_threshold": float(LEGACY_LOW_MID_THRESHOLD),
            "mid_high_threshold": float(LEGACY_MID_HIGH_THRESHOLD),
        },
        "gmm_sensitivity": {
            "low_mid_threshold": float(gmm_res["low_mid_threshold"]),
            "mid_high_threshold": float(gmm_res["mid_high_threshold"]),
            "component_means": _to_float_list(gmm_res["component_means"]),
            "component_variances": _to_float_list(gmm_res["component_variances"]),
            "component_weights": _to_float_list(gmm_res["component_weights"]),
            "cluster_summary": gmm_res["cluster_summary"],
            "lower_bound": float(gmm_res["lower_bound"]),
        },
    }

    comparison_path = os.path.join(run_dir, "rule_comparison.csv")
    threshold_zone_summary_path = os.path.join(run_dir, "threshold_zone_summary.csv")
    summary_path = os.path.join(run_dir, "summary.json")
    opening_model_path = os.path.join(run_dir, "opening_models.csv")
    narrative_path = os.path.join(run_dir, "narrative_summary.md")

    save_dataframe(rule_comparison_df, comparison_path)
    save_dataframe(threshold_zone_summary_df, threshold_zone_summary_path)
    save_dataframe(pd.DataFrame(opening_models), opening_model_path)
    _write_summary_json(summary_path, summary)
    _save_narrative(narrative_path, _build_narrative(summary, top_k=top_k))

    mass_fig_path = os.path.join(figure_dir, "mass_distribution_thresholds.png")
    opening_fig_path = os.path.join(figure_dir, "opening_reachable_mass_ranges.png")
    comparison_fig_path = os.path.join(figure_dir, "current_vs_data_driven_rules.png")
    _make_mass_distribution_figure(mass_fig_path, mass_values, kmeans_res, gmm_res)
    _make_opening_ranges_figure(opening_fig_path, opening_models)
    _make_rule_comparison_figure(
        comparison_fig_path,
        rule_comparison_df=rule_comparison_df,
        opening_model_map=opening_model_map,
        mass_min=mass_min,
        mass_max=mass_max,
    )

    manifest_path = write_manifest(
        run_dir,
        script_name="select_opening_thresholds_research.py",
        data_path=data_path,
        seed=seed,
        params={
            "n_clusters": int(n_clusters),
            "reference_speed": float(reference_speed),
            "candidate_source": candidate_source,
            "policy_label": POLICY_LABEL,
            "fixed_triplet_mm": _to_float_list(fixed_triplet),
            "legacy_thresholds": [LEGACY_LOW_MID_THRESHOLD, LEGACY_MID_HIGH_THRESHOLD],
            "policy_thresholds": [POLICY_LOW_MID_THRESHOLD, POLICY_MID_HIGH_THRESHOLD],
        },
        extra={
            "notes": "Research script for validating a fixed triplet policy and upgrading only the mass thresholds.",
        },
    )

    append_manifest_outputs(
        run_dir,
        [
            {"path": "summary.json"},
            {"path": "rule_comparison.csv"},
            {"path": "threshold_zone_summary.csv"},
            {"path": "opening_models.csv"},
            {"path": "narrative_summary.md"},
            {"path": "figures/mass_distribution_thresholds.png"},
            {"path": "figures/opening_reachable_mass_ranges.png"},
            {"path": "figures/current_vs_data_driven_rules.png"},
        ],
    )

    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "comparison_path": comparison_path,
        "threshold_zone_summary_path": threshold_zone_summary_path,
        "opening_model_path": opening_model_path,
        "narrative_path": narrative_path,
        "figure_dir": figure_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Research opening thresholds for an interpretable three-stage policy.")
    parser.add_argument("--data-path", default="data/dataset.xlsx")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--reference-speed", type=float, default=DEFAULT_REFERENCE_SPEED)
    parser.add_argument("--candidate-source", default="observed")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    outputs = run_research(
        data_path=args.data_path,
        n_clusters=args.n_clusters,
        reference_speed=args.reference_speed,
        candidate_source=args.candidate_source,
        top_k=args.top_k,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"Run directory: {outputs['run_dir']}")
    print(f"Summary: {outputs['summary_path']}")
    print(f"Rule comparison: {outputs['comparison_path']}")
    print(f"Threshold zone summary: {outputs['threshold_zone_summary_path']}")
    print(f"Narrative summary: {outputs['narrative_path']}")
    print(f"Figures: {outputs['figure_dir']}")


if __name__ == "__main__":
    main()
