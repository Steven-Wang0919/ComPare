# -*- coding: utf-8 -*-
"""
单文件科研实验脚本：自动搜索 low / mid / high 三档开度阈值

使用方式（PyCharm 直接点击运行）：
1. 修改下方 EXPERIMENT_CONFIG 中的数据路径与输出目录；
2. 点击 Run；
3. 程序会在终端打印最优结果，并将完整结果保存到 output_dir。

输出文件：
- opening_models.csv         每个训练开度对应的线性拟合参数
- candidate_scores.csv       每个候选整数开度在 low/mid/high 三种角色下的评分
- triplet_ranking.csv        所有 low<mid<high 三元组的总排序
- summary.json               最终实验摘要
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# 实验配置：在 PyCharm 里直接修改这里即可
# ============================================================================
EXPERIMENT_CONFIG = {
    "data_path": "data/dataset.xlsx",          # Excel 数据路径
    "output_dir": "runs/opening_thresholds",  # 结果保存目录
    "reference_speed": None,                    # 参考转速；None=自动取 speed 中位数
    "fixed_min_distance": None,                 # 固定插值候选开度与训练开度的最小距离；None=自动搜索
    "include_training_openings": True,          # True=20/25/30/... 这些训练开度也参与候选评估
    "top_k": 20,                                # summary.json 中保留前 k 个三元组
    "random_seed": 42,
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

    n_clusters: int = 3
    kmeans_restarts: int = 16
    kmeans_max_iter: int = 200

    # 单候选评分权重
    w_fit: float = 0.50
    w_coverage: float = 0.25
    w_distance: float = 0.15
    w_order: float = 0.10

    # 三元组附加评分
    w_spacing: float = 0.08
    w_balance: float = 0.04
    w_distance_bonus: float = 0.03

    def validate(self) -> None:
        if self.n_clusters != 3:
            raise ValueError("当前脚本固定为 3 类质量区间（low/mid/high）。")
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0。")
        if self.kmeans_restarts <= 0 or self.kmeans_max_iter <= 0:
            raise ValueError("KMeans 参数必须为正数。")
        if self.fixed_min_distance is not None and self.fixed_min_distance < 0:
            raise ValueError("fixed_min_distance 不能小于 0。")


# ============================================================================
# 数据与拟合
# ============================================================================
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


def get_reference_speed(df: pd.DataFrame, cfg: SearchConfig) -> float:
    return float(cfg.reference_speed) if cfg.reference_speed is not None else float(df["speed"].median())


def interpolate_params(opening: float, models: pd.DataFrame) -> Tuple[float, float]:
    xp = models["opening"].to_numpy(dtype=float)
    if opening < xp.min() or opening > xp.max():
        raise ValueError(f"开度 {opening} 超出训练开度范围，无法插值。")
    slope = float(np.interp(opening, xp, models["slope"].to_numpy(dtype=float)))
    intercept = float(np.interp(opening, xp, models["intercept"].to_numpy(dtype=float)))
    return slope, intercept


def predict_mass(opening: float, speed: float, models: pd.DataFrame) -> float:
    slope, intercept = interpolate_params(opening, models)
    return float(slope * speed + intercept)


# ============================================================================
# 自动学习质量阈值（1D KMeans）
# ============================================================================
def kmeans_1d(values: np.ndarray, n_clusters: int, restarts: int, max_iter: int, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < n_clusters:
        raise ValueError("样本量不足，无法进行 1D KMeans。")

    rng = np.random.default_rng(seed)
    init_percentiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
    best_centers, best_labels, best_inertia = None, None, np.inf

    for restart in range(restarts):
        if restart == 0:
            centers = np.percentile(values, init_percentiles).astype(float)
        else:
            qs = np.sort(rng.uniform(0, 100, size=n_clusters))
            centers = np.percentile(values, qs).astype(float)

        for _ in range(max_iter):
            distances = np.abs(values[:, None] - centers[None, :])
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for k in range(n_clusters):
                cluster_values = values[labels == k]
                if len(cluster_values) > 0:
                    new_centers[k] = float(cluster_values.mean())
            if np.allclose(new_centers, centers, atol=1e-8, rtol=0.0):
                centers = new_centers
                break
            centers = new_centers

        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum((values - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_centers = centers.copy()
            best_labels = labels.copy()
            best_inertia = inertia

    assert best_centers is not None and best_labels is not None
    order = np.argsort(best_centers)
    centers_sorted = best_centers[order]
    remap = {old: new for new, old in enumerate(order)}
    labels_sorted = np.array([remap[int(label)] for label in best_labels], dtype=int)
    return centers_sorted, labels_sorted, best_inertia


def learn_mass_thresholds(df: pd.DataFrame, cfg: SearchConfig) -> Dict[str, Any]:
    masses = df["mass"].to_numpy(dtype=float)
    centers, labels, inertia = kmeans_1d(
        masses,
        n_clusters=cfg.n_clusters,
        restarts=cfg.kmeans_restarts,
        max_iter=cfg.kmeans_max_iter,
        seed=cfg.random_seed,
    )

    t1 = float(0.5 * (centers[0] + centers[1]))
    t2 = float(0.5 * (centers[1] + centers[2]))

    cluster_stats = []
    for k in range(3):
        vals = masses[labels == k]
        cluster_stats.append({
            "cluster": int(k),
            "center": float(centers[k]),
            "count": int(len(vals)),
            "mass_min": float(vals.min()),
            "mass_max": float(vals.max()),
        })

    return {
        "low_mid_threshold": t1,
        "mid_high_threshold": t2,
        "cluster_centers": centers.tolist(),
        "cluster_stats": cluster_stats,
        "kmeans_inertia": float(inertia),
    }


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


# ============================================================================
# 候选开度与评分
# ============================================================================
def generate_candidates(
    df: pd.DataFrame,
    min_distance: int,
    internal_only: bool,
    include_training_openings: bool = True,
) -> List[int]:
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


def zone_closeness(value: float, zone: Tuple[float, float]) -> float:
    z0, z1 = zone
    center = 0.5 * (z0 + z1)
    half_width = max(1e-9, 0.5 * (z1 - z0))
    return float(max(0.0, min(1.0, 1.0 - abs(value - center) / half_width)))


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


def score_candidates(
    df: pd.DataFrame,
    models: pd.DataFrame,
    thresholds: Dict[str, Any],
    min_distance: int,
    cfg: SearchConfig,
) -> pd.DataFrame:
    train_openings = np.sort(df["opening"].unique().astype(float))
    candidates = generate_candidates(df, min_distance, cfg.require_internal_range, cfg.include_training_openings)
    if len(candidates) < 3:
        raise ValueError("候选开度不足 3 个。")

    ref_speed = get_reference_speed(df, cfg)
    speed_min = float(df["speed"].min())
    speed_max = float(df["speed"].max())
    zones = build_mass_zones(df, thresholds)
    max_spacing = max(1.0, float(np.max(np.diff(train_openings)))) if len(train_openings) >= 2 else 1.0

    rows: List[Dict[str, Any]] = []
    for opening in candidates:
        est_ref = predict_mass(opening, ref_speed, models)
        est_min = predict_mass(opening, speed_min, models)
        est_max = predict_mass(opening, speed_max, models)
        est_interval = (min(est_min, est_max), max(est_min, est_max))

        nearest = float(np.min(np.abs(train_openings - opening)))
        is_training_opening = bool(np.isclose(train_openings, opening).any())
        distance_score = 0.0 if is_training_opening else min(1.0, nearest / max_spacing)

        role_scores: Dict[str, float] = {}
        role_coverage: Dict[str, float] = {}
        role_order_scores: Dict[str, float] = {}
        for role, zone in zones.items():
            fit = zone_closeness(est_ref, zone)
            coverage = overlap_ratio(est_interval, zone)
            order = order_score(est_ref, role, thresholds)
            score = (
                cfg.w_fit * fit
                + cfg.w_coverage * coverage
                + cfg.w_distance * distance_score
                + cfg.w_order * order
            )
            role_scores[role] = float(score)
            role_coverage[role] = float(coverage)
            role_order_scores[role] = float(order)

        best_role = max(role_scores, key=role_scores.get)
        rows.append({
            "opening": int(opening),
            "selected_min_distance": int(min_distance),
            "estimated_mass_ref": float(est_ref),
            "estimated_mass_min": float(est_interval[0]),
            "estimated_mass_max": float(est_interval[1]),
            "min_distance_to_train": float(nearest),
            "distance_score": float(distance_score),
            "low_score": float(role_scores["low"]),
            "mid_score": float(role_scores["mid"]),
            "high_score": float(role_scores["high"]),
            "low_coverage": float(role_coverage["low"]),
            "mid_coverage": float(role_coverage["mid"]),
            "high_coverage": float(role_coverage["high"]),
            "best_role": best_role,
            "is_training_opening": is_training_opening,
            "best_role_order_score": float(role_order_scores[best_role]),
        })

    return pd.DataFrame(rows).sort_values("opening").reset_index(drop=True)


# ============================================================================
# 三元组搜索
# ============================================================================
def spacing_score(low: int, mid: int, high: int, cand_min: int, cand_max: int) -> float:
    span = max(1.0, float(cand_max - cand_min))
    return float(max(0.0, min(1.0, ((mid - low) + (high - mid)) / span)))


def balance_score(low: int, mid: int, high: int) -> float:
    g1 = float(mid - low)
    g2 = float(high - mid)
    return float(1.0 - abs(g1 - g2) / max(1e-9, g1 + g2))


def rank_triplets(candidate_scores: pd.DataFrame, selected_min_distance: int, feasible_distances: List[int], cfg: SearchConfig) -> pd.DataFrame:
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
                space = spacing_score(int(low), int(mid), int(high), cand_min, cand_max)
                balance = balance_score(int(low), int(mid), int(high))
                total = (
                    float(low_row["low_score"])
                    + float(mid_row["mid_score"])
                    + float(high_row["high_score"])
                    + cfg.w_spacing * space
                    + cfg.w_balance * balance
                    + cfg.w_distance_bonus * distance_bonus
                )
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
                    "spacing_score": float(space),
                    "balance_score": float(balance),
                    "distance_bonus": float(distance_bonus),
                    "total_score": float(total),
                })

    if not rows:
        raise ValueError("没有找到满足 low < mid < high 的候选三元组。")

    ranking = pd.DataFrame(rows).sort_values(
        ["total_score", "selected_min_distance", "high_opening", "mid_opening", "low_opening"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


# ============================================================================
# 实验入口
# ============================================================================
def run_experiment(data_path: str | Path, output_dir: str | Path, cfg: SearchConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or SearchConfig()
    cfg.validate()

    df = load_data(data_path, cfg)
    models = fit_opening_models(df)
    thresholds = learn_mass_thresholds(df, cfg)
    feasible_distances = get_feasible_distances(df, cfg)

    candidate_tables = []
    triplet_tables = []
    for d in feasible_distances:
        candidates = score_candidates(df, models, thresholds, d, cfg)
        candidate_tables.append(candidates)
        triplets = rank_triplets(candidates, d, feasible_distances, cfg)
        triplet_tables.append(triplets)

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

    print("\n" + "=" * 78)
    print("开度阈值自动搜索实验")
    print("=" * 78)
    print(f"数据文件           : {summary['data_path']}")
    print(f"参考转速           : {summary['reference_speed']:.2f} r/min")
    print(f"训练开度           : {summary['train_openings']}")
    print(f"low / mid 阈值     : {th['low_mid_threshold']:.2f} g/min")
    print(f"mid / high 阈值    : {th['mid_high_threshold']:.2f} g/min")
    print(f"可行最小距离集合   : {summary['feasible_min_distances']}")
    print("-" * 78)
    print("最优三元组")
    print(f"low_opening        : {best['low_opening']} mm")
    print(f"mid_opening        : {best['mid_opening']} mm")
    print(f"high_opening       : {best['high_opening']} mm")
    print(f"selected_distance  : {best['selected_min_distance']} mm")
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
