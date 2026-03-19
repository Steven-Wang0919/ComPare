# -*- coding: utf-8 -*-
"""
学术增强版单文件科研实验脚本：自动搜索 low / mid / high 三档开度阈值

增强点：
1. GMM（Gaussian Mixture Model）替代 1D KMeans，得到 soft boundary；
2. PCHIP（保形分段三次 Hermite 插值）替代线性插值；
3. 熵权法替代固定权重，分别用于：
   - 单候选的角色评分（fit / coverage / distance / order）
   - 三元组总评分（low / mid / high / spacing / balance / distance_bonus）

使用方式（PyCharm 直接点击运行）：
1. 修改下方 EXPERIMENT_CONFIG 中的数据路径与输出目录；
2. 点击 Run；
3. 程序会在终端打印最优结果，并将完整结果保存到 output_dir。
"""

from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = '1'
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.mixture import GaussianMixture


EXPERIMENT_CONFIG = {
    "data_path": "data/dataset.xlsx",
    "output_dir": "runs/opening_thresholds_academic",
    "reference_speed": None,
    "fixed_min_distance": None,
    "include_training_openings": True,
    "top_k": 20,
    "random_seed": 42,
    "gmm_reg_covar": 1e-6,
}


@dataclass
class SearchConfig:
    opening_col: str = "排肥口开度（mm）"
    speed_col: str = "排肥轴转速（r/min）"
    mass_col: str = "实际排肥质量（g/min）"

    reference_speed: float | None = None
    fixed_min_distance: int | None = None
    require_internal_range: bool = True
    include_training_openings: bool = True
    top_k: int = 20
    random_seed: int = 42
    gmm_reg_covar: float = 1e-6

    n_components: int = 3
    eps: float = 1e-12

    def validate(self) -> None:
        if self.n_components != 3:
            raise ValueError("当前脚本固定为 3 类质量区间（low/mid/high）。")
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0。")
        if self.fixed_min_distance is not None and self.fixed_min_distance < 0:
            raise ValueError("fixed_min_distance 不能小于 0。")
        if self.gmm_reg_covar <= 0:
            raise ValueError("gmm_reg_covar 必须为正数。")


def load_data(path: str | Path, cfg: SearchConfig) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    df = pd.read_excel(path)
    required = {cfg.opening_col, cfg.speed_col, cfg.mass_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必要列: {sorted(missing)}")

    df = df[[cfg.opening_col, cfg.speed_col, cfg.mass_col]].copy()
    df.columns = ["opening", "speed", "mass"]
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("数据为空，无法执行搜索。")
    return df


def fit_opening_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for opening, sub in df.groupby("opening"):
        x = sub["speed"].to_numpy(dtype=float)
        y = sub["mass"].to_numpy(dtype=float)
        if len(np.unique(x)) < 2:
            raise ValueError(f"开度 {opening} 的转速取值不足，无法拟合线性模型。")

        slope, intercept = np.polyfit(x, y, deg=1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

        rows.append({
            "opening": float(opening),
            "slope": float(slope),
            "intercept": float(intercept),
            "speed_min": float(x.min()),
            "speed_max": float(x.max()),
            "mass_min": float(y.min()),
            "mass_max": float(y.max()),
            "r2": float(r2) if not np.isnan(r2) else np.nan,
        })

    models = pd.DataFrame(rows).sort_values("opening").reset_index(drop=True)
    if len(models) < 2:
        raise ValueError("训练开度不足 2 个，无法做插值搜索。")
    return models


class ParamInterpolator:
    def __init__(self, models: pd.DataFrame):
        xp = models["opening"].to_numpy(dtype=float)
        self.x_min = float(xp.min())
        self.x_max = float(xp.max())
        self.slope_interp = PchipInterpolator(xp, models["slope"].to_numpy(dtype=float), extrapolate=False)
        self.intercept_interp = PchipInterpolator(xp, models["intercept"].to_numpy(dtype=float), extrapolate=False)

    def params(self, opening: float) -> Tuple[float, float]:
        if opening < self.x_min or opening > self.x_max:
            raise ValueError(f"开度 {opening} 超出训练开度范围，无法插值。")
        slope = float(self.slope_interp(opening))
        intercept = float(self.intercept_interp(opening))
        return slope, intercept

    def predict(self, opening: float, speed: float) -> float:
        slope, intercept = self.params(opening)
        return float(slope * speed + intercept)


def get_reference_speed(df: pd.DataFrame, cfg: SearchConfig) -> float:
    return float(cfg.reference_speed) if cfg.reference_speed is not None else float(df["speed"].median())


def normal_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    coef = 1.0 / np.sqrt(2.0 * np.pi * var)
    return coef * np.exp(-0.5 * ((x - mean) ** 2) / var)


def find_gmm_boundary(gmm: GaussianMixture, mean_a: float, mean_b: float) -> float:
    means = gmm.means_.reshape(-1)
    variances = gmm.covariances_.reshape(-1)
    weights = gmm.weights_.reshape(-1)
    idx_a = int(np.argmin(np.abs(means - mean_a)))
    idx_b = int(np.argmin(np.abs(means - mean_b)))
    left, right = sorted([mean_a, mean_b])
    grid = np.linspace(left, right, 2000)
    pa = weights[idx_a] * normal_pdf(grid, means[idx_a], variances[idx_a])
    pb = weights[idx_b] * normal_pdf(grid, means[idx_b], variances[idx_b])
    diff = pa - pb
    change_idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(change_idx) == 0:
        return float(0.5 * (mean_a + mean_b))
    i = int(change_idx[0])
    x0, x1 = float(grid[i]), float(grid[i + 1])
    y0, y1 = float(diff[i]), float(diff[i + 1])
    if abs(y1 - y0) < 1e-12:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def learn_mass_thresholds(df: pd.DataFrame, cfg: SearchConfig) -> Dict[str, Any]:
    masses = df["mass"].to_numpy(dtype=float).reshape(-1, 1)
    if len(masses) < cfg.n_components:
        raise ValueError("样本量不足，无法进行 GMM 分档。")

    gmm = GaussianMixture(
        n_components=cfg.n_components,
        covariance_type="full",
        reg_covar=cfg.gmm_reg_covar,
        random_state=cfg.random_seed,
        n_init=10,
        max_iter=500,
    )
    gmm.fit(masses)

    raw_means = gmm.means_.reshape(-1)
    order = np.argsort(raw_means)
    means = raw_means[order]
    variances = gmm.covariances_.reshape(-1)[order]
    weights = gmm.weights_.reshape(-1)[order]

    t1 = find_gmm_boundary(gmm, float(means[0]), float(means[1]))
    t2 = find_gmm_boundary(gmm, float(means[1]), float(means[2]))

    responsibilities = gmm.predict_proba(masses)[:, order]
    labels = np.argmax(responsibilities, axis=1)
    cluster_stats = []
    mass_values = masses.reshape(-1)
    for k in range(3):
        vals = mass_values[labels == k]
        cluster_stats.append({
            "component": int(k),
            "mean": float(means[k]),
            "variance": float(variances[k]),
            "weight": float(weights[k]),
            "count": int(len(vals)),
            "mass_min": float(vals.min()) if len(vals) else np.nan,
            "mass_max": float(vals.max()) if len(vals) else np.nan,
        })

    return {
        "low_mid_threshold": float(t1),
        "mid_high_threshold": float(t2),
        "component_means": means.tolist(),
        "component_variances": variances.tolist(),
        "component_weights": weights.tolist(),
        "cluster_stats": cluster_stats,
        "gmm_lower_bound": float(gmm.lower_bound_),
    }


class GMMScorer:
    def __init__(self, thresholds: Dict[str, Any]):
        self.means = np.asarray(thresholds["component_means"], dtype=float)
        self.vars = np.asarray(thresholds["component_variances"], dtype=float)
        self.weights = np.asarray(thresholds["component_weights"], dtype=float)
        self.t1 = float(thresholds["low_mid_threshold"])
        self.t2 = float(thresholds["mid_high_threshold"])
        self.roles = ["low", "mid", "high"]

    def posterior(self, value: float) -> Dict[str, float]:
        x = np.array([float(value)], dtype=float)
        comps = self.weights * np.array([
            normal_pdf(x, self.means[i], self.vars[i])[0] for i in range(3)
        ])
        denom = float(np.sum(comps))
        if denom <= 1e-12:
            probs = np.ones(3, dtype=float) / 3.0
        else:
            probs = comps / denom
        return {role: float(probs[i]) for i, role in enumerate(self.roles)}


def build_mass_zones(df: pd.DataFrame, thresholds: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    data_min = float(df["mass"].min())
    data_max = float(df["mass"].max())
    t1 = float(thresholds["low_mid_threshold"])
    t2 = float(thresholds["mid_high_threshold"])
    return {
        "low": (data_min, t1),
        "mid": (t1, t2),
        "high": (t2, data_max),
    }


def generate_candidates(df: pd.DataFrame, min_distance: int, internal_only: bool, include_training_openings: bool = True) -> List[int]:
    train_openings = np.sort(df["opening"].unique().astype(float))
    lo, hi = int(train_openings.min()), int(train_openings.max())
    search_range = range(lo + 1, hi) if internal_only else range(lo, hi + 1)

    candidates = set()
    for opening in search_range:
        nearest = float(np.min(np.abs(train_openings - opening)))
        is_training = np.isclose(train_openings, opening).any()
        if is_training:
            if include_training_openings:
                candidates.add(int(opening))
            continue
        if nearest >= min_distance:
            candidates.add(int(opening))

    if include_training_openings:
        for opening in train_openings:
            opening_int = int(round(float(opening)))
            if internal_only and not (lo <= opening_int <= hi):
                continue
            candidates.add(opening_int)

    return sorted(candidates)


def get_feasible_distances(df: pd.DataFrame, cfg: SearchConfig) -> List[int]:
    if cfg.fixed_min_distance is not None:
        candidates = generate_candidates(df, int(cfg.fixed_min_distance), cfg.require_internal_range, cfg.include_training_openings)
        if len(candidates) < 3:
            raise ValueError("当前 fixed_min_distance 下候选开度不足 3 个。")
        return [int(cfg.fixed_min_distance)]

    train_openings = np.sort(df["opening"].unique().astype(float))
    max_testable = max(1, int(train_openings.max() - train_openings.min()))
    feasible = []
    for d in range(1, max_testable + 1):
        if len(generate_candidates(df, d, cfg.require_internal_range, cfg.include_training_openings)) >= 3:
            feasible.append(d)
    if not feasible:
        raise ValueError("没有任何可行的最小距离能产生至少 3 个候选开度。")
    return feasible


def overlap_ratio(interval_a: Tuple[float, float], interval_b: Tuple[float, float]) -> float:
    a0, a1 = interval_a
    b0, b1 = interval_b
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-9, b1 - b0)
    return float(overlap / denom)


def order_score(value: float, role: str, thresholds: Dict[str, Any]) -> float:
    t1 = float(thresholds["low_mid_threshold"])
    t2 = float(thresholds["mid_high_threshold"])
    margin = max(1.0, t2 - t1)
    if role == "low":
        return float(max(0.0, min(1.0, (t1 - value) / margin + 0.5)))
    if role == "mid":
        center = 0.5 * (t1 + t2)
        half_width = max(1.0, 0.5 * (t2 - t1))
        return float(max(0.0, min(1.0, 1.0 - abs(value - center) / half_width)))
    return float(max(0.0, min(1.0, (value - t2) / margin + 0.5)))


def normalize_benefit_columns(df: pd.DataFrame, columns: List[str], eps: float = 1e-12) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        vals = out[col].to_numpy(dtype=float)
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if abs(vmax - vmin) <= eps:
            out[col] = 1.0
        else:
            out[col] = (vals - vmin) / (vmax - vmin) + eps
    return out


def compute_entropy_weights(df: pd.DataFrame, columns: List[str], eps: float = 1e-12) -> Dict[str, float]:
    norm_df = normalize_benefit_columns(df[columns], columns, eps=eps)
    X = norm_df[columns].to_numpy(dtype=float)
    P = X / np.clip(X.sum(axis=0, keepdims=True), eps, None)
    n = X.shape[0]
    if n <= 1:
        return {col: 1.0 / len(columns) for col in columns}
    k = 1.0 / np.log(n)
    entropy = -k * np.sum(P * np.log(np.clip(P, eps, None)), axis=0)
    divergence = 1.0 - entropy
    if float(np.sum(divergence)) <= eps:
        weights = np.ones(len(columns), dtype=float) / len(columns)
    else:
        weights = divergence / np.sum(divergence)
    return {col: float(weights[i]) for i, col in enumerate(columns)}


def score_candidates(
    df: pd.DataFrame,
    interpolator: ParamInterpolator,
    thresholds: Dict[str, Any],
    min_distance: int,
    cfg: SearchConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    train_openings = np.sort(df["opening"].unique().astype(float))
    candidates = generate_candidates(df, min_distance, cfg.require_internal_range, cfg.include_training_openings)
    if len(candidates) < 3:
        raise ValueError("候选开度不足 3 个。")

    ref_speed = get_reference_speed(df, cfg)
    speed_min = float(df["speed"].min())
    speed_max = float(df["speed"].max())
    zones = build_mass_zones(df, thresholds)
    scorer = GMMScorer(thresholds)
    max_spacing = max(1.0, float(np.max(np.diff(train_openings)))) if len(train_openings) >= 2 else 1.0

    rows: List[Dict[str, Any]] = []
    for opening in candidates:
        est_ref = interpolator.predict(opening, ref_speed)
        est_min = interpolator.predict(opening, speed_min)
        est_max = interpolator.predict(opening, speed_max)
        est_interval = (min(est_min, est_max), max(est_min, est_max))
        nearest = float(np.min(np.abs(train_openings - opening)))
        is_training_opening = bool(np.isclose(train_openings, opening).any())
        distance_score = 0.0 if is_training_opening else min(1.0, nearest / max_spacing)
        role_posterior = scorer.posterior(est_ref)

        row: Dict[str, Any] = {
            "opening": int(opening),
            "selected_min_distance": int(min_distance),
            "estimated_mass_ref": float(est_ref),
            "estimated_mass_min": float(est_interval[0]),
            "estimated_mass_max": float(est_interval[1]),
            "min_distance_to_train": float(nearest),
            "distance_score": float(distance_score),
            "is_training_opening": is_training_opening,
        }
        for role, zone in zones.items():
            fit_prob = float(role_posterior[role])
            coverage = overlap_ratio(est_interval, zone)
            order = order_score(est_ref, role, thresholds)
            row[f"{role}_fit_prob"] = fit_prob
            row[f"{role}_coverage"] = float(coverage)
            row[f"{role}_distance_metric"] = float(distance_score)
            row[f"{role}_order_metric"] = float(order)
        rows.append(row)

    table = pd.DataFrame(rows)
    role_weights: Dict[str, Dict[str, float]] = {}
    for role in ["low", "mid", "high"]:
        metric_cols = [
            f"{role}_fit_prob",
            f"{role}_coverage",
            f"{role}_distance_metric",
            f"{role}_order_metric",
        ]
        weights = compute_entropy_weights(table, metric_cols, eps=cfg.eps)
        role_weights[role] = weights
        norm_role = normalize_benefit_columns(table[metric_cols], metric_cols, eps=cfg.eps)
        table[f"{role}_score"] = 0.0
        for col in metric_cols:
            table[f"{role}_score"] += norm_role[col] * weights[col]

    score_cols = ["low_score", "mid_score", "high_score"]
    table[score_cols] = table[score_cols].astype(float)
    table["best_role"] = table[score_cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    table["best_role_order_score"] = table.apply(lambda r: float(r[f"{r['best_role']}_order_metric"]), axis=1)
    table = table.sort_values("opening").reset_index(drop=True)
    return table, role_weights


def spacing_score(low: int, mid: int, high: int, cand_min: int, cand_max: int) -> float:
    span = max(1.0, float(cand_max - cand_min))
    return float(max(0.0, min(1.0, ((mid - low) + (high - mid)) / span)))


def balance_score(low: int, mid: int, high: int) -> float:
    g1 = float(mid - low)
    g2 = float(high - mid)
    return float(1.0 - abs(g1 - g2) / max(1e-9, g1 + g2))


def build_triplets(candidate_scores: pd.DataFrame, selected_min_distance: int, feasible_distances: List[int]) -> pd.DataFrame:
    table = candidate_scores.sort_values("opening").reset_index(drop=True)
    openings = table["opening"].tolist()
    lookup = {int(row["opening"]): row for _, row in table.iterrows()}
    cand_min, cand_max = int(min(openings)), int(max(openings))
    distance_bonus = selected_min_distance / max(1.0, float(max(feasible_distances)))

    rows = []
    for i, low in enumerate(openings):
        for j in range(i + 1, len(openings)):
            mid = openings[j]
            for k in range(j + 1, len(openings)):
                high = openings[k]
                low_row = lookup[int(low)]
                mid_row = lookup[int(mid)]
                high_row = lookup[int(high)]
                rows.append({
                    "low_opening": int(low),
                    "mid_opening": int(mid),
                    "high_opening": int(high),
                    "selected_min_distance": int(selected_min_distance),
                    "low_estimated_mass": float(low_row["estimated_mass_ref"]),
                    "mid_estimated_mass": float(mid_row["estimated_mass_ref"]),
                    "high_estimated_mass": float(high_row["estimated_mass_ref"]),
                    "low_role_score": float(low_row["low_score"]),
                    "mid_role_score": float(mid_row["mid_score"]),
                    "high_role_score": float(high_row["high_score"]),
                    "spacing_score": float(spacing_score(int(low), int(mid), int(high), cand_min, cand_max)),
                    "balance_score": float(balance_score(int(low), int(mid), int(high))),
                    "distance_bonus": float(distance_bonus),
                })

    if not rows:
        raise ValueError("没有找到满足 low < mid < high 的候选三元组。")
    return pd.DataFrame(rows)


def rank_triplets(triplets: pd.DataFrame, cfg: SearchConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    metric_cols = [
        "low_role_score",
        "mid_role_score",
        "high_role_score",
        "spacing_score",
        "balance_score",
        "distance_bonus",
    ]
    weights = compute_entropy_weights(triplets, metric_cols, eps=cfg.eps)
    norm_triplets = normalize_benefit_columns(triplets[metric_cols], metric_cols, eps=cfg.eps)
    ranking = triplets.copy()
    ranking["total_score"] = 0.0
    for col in metric_cols:
        ranking["total_score"] += norm_triplets[col] * weights[col]
    ranking = ranking.sort_values(
        ["total_score", "selected_min_distance", "high_opening", "mid_opening", "low_opening"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking, weights


def run_experiment(data_path: str | Path, output_dir: str | Path, cfg: SearchConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or SearchConfig()
    cfg.validate()

    df = load_data(data_path, cfg)
    models = fit_opening_models(df)
    interpolator = ParamInterpolator(models)
    thresholds = learn_mass_thresholds(df, cfg)
    feasible_distances = get_feasible_distances(df, cfg)

    candidate_tables = []
    triplet_tables = []
    candidate_role_weights: Dict[str, Dict[str, float]] = {}
    triplet_entropy_weights: Dict[str, float] = {}

    for d in feasible_distances:
        candidates, role_weights = score_candidates(df, interpolator, thresholds, d, cfg)
        candidate_tables.append(candidates)
        for role, weights in role_weights.items():
            candidate_role_weights[f"distance_{d}_{role}"] = weights
        triplets = build_triplets(candidates, d, feasible_distances)
        ranking, triplet_weights = rank_triplets(triplets, cfg)
        triplet_tables.append(ranking)
        triplet_entropy_weights[f"distance_{d}"] = triplet_weights

    candidate_scores = pd.concat(candidate_tables, ignore_index=True)
    triplet_ranking = pd.concat(triplet_tables, ignore_index=True)
    triplet_ranking = triplet_ranking.sort_values(
        ["total_score", "selected_min_distance", "high_opening", "mid_opening", "low_opening"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    triplet_ranking["rank"] = np.arange(1, len(triplet_ranking) + 1)

    best = triplet_ranking.iloc[0].to_dict()
    summary = {
        "config": asdict(cfg),
        "data_path": str(data_path),
        "reference_speed": get_reference_speed(df, cfg),
        "train_openings": sorted(df["opening"].unique().astype(float).tolist()),
        "learned_thresholds": thresholds,
        "mass_zones": build_mass_zones(df, thresholds),
        "feasible_min_distances": feasible_distances,
        "candidate_role_entropy_weights": candidate_role_weights,
        "triplet_entropy_weights": triplet_entropy_weights,
        "best_triplet": best,
        "top_triplets": triplet_ranking.head(cfg.top_k).to_dict(orient="records"),
    }

    save_results(output_dir, models, candidate_scores, triplet_ranking, summary)
    summary["output_dir"] = str(Path(output_dir))
    return summary


def save_results(output_dir: str | Path, models: pd.DataFrame, candidate_scores: pd.DataFrame, triplet_ranking: pd.DataFrame, summary: Dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models.to_csv(output_dir / "opening_models.csv", index=False, encoding="utf-8-sig")
    candidate_scores.to_csv(output_dir / "candidate_scores.csv", index=False, encoding="utf-8-sig")
    triplet_ranking.to_csv(output_dir / "triplet_ranking.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_report(summary: Dict[str, Any]) -> None:
    best = summary["best_triplet"]
    th = summary["learned_thresholds"]
    print("=" * 78)
    print("开度阈值自动搜索实验（学术增强版：GMM + PCHIP + 熵权法）")
    print("=" * 78)
    print(f"数据文件           : {summary['data_path']}")
    print(f"参考转速           : {summary['reference_speed']:.2f} r/min")
    print(f"训练开度           : {summary['train_openings']}")
    print(f"low / mid 阈值     : {th['low_mid_threshold']:.2f} g/min")
    print(f"mid / high 阈值    : {th['mid_high_threshold']:.2f} g/min")
    print(f"可行最小距离集合   : {summary['feasible_min_distances']}")
    print("-" * 78)
    print("最优三元组")
    print(f"low_opening        : {best['low_opening']:.1f} mm")
    print(f"mid_opening        : {best['mid_opening']:.1f} mm")
    print(f"high_opening       : {best['high_opening']:.1f} mm")
    print(f"selected_distance  : {best['selected_min_distance']:.1f} mm")
    print(f"estimated_mass     : [{best['low_estimated_mass']:.2f}, {best['mid_estimated_mass']:.2f}, {best['high_estimated_mass']:.2f}] g/min")
    print(f"total_score        : {best['total_score']:.6f}")
    print("-" * 78)
    print(f"结果目录           : {summary['output_dir']}")
    print("=" * 78)


def build_config(settings: Dict[str, Any]) -> SearchConfig:
    return SearchConfig(
        reference_speed=settings.get("reference_speed"),
        fixed_min_distance=settings.get("fixed_min_distance"),
        include_training_openings=bool(settings.get("include_training_openings", True)),
        top_k=int(settings.get("top_k", 20)),
        random_seed=int(settings.get("random_seed", 42)),
        gmm_reg_covar=float(settings.get("gmm_reg_covar", 1e-6)),
    )


def main() -> None:
    cfg = build_config(EXPERIMENT_CONFIG)
    summary = run_experiment(
        data_path=EXPERIMENT_CONFIG["data_path"],
        output_dir=EXPERIMENT_CONFIG["output_dir"],
        cfg=cfg,
    )
    print_report(summary)


if __name__ == "__main__":
    main()
