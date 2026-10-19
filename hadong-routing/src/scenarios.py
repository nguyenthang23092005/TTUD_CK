"""Giai đoạn 3 — sinh tập truy vấn cố định (train/val/test tách theo cặp O–D).

Mỗi truy vấn: (origin, dest, giờ xuất phát (phút), sigma nhiễu, seed nhiễu, danh sách sự cố).
V_real = enforce_fifo(V_base * eps), eps ~ LogNormal(0, sigma) theo seed của truy vấn; cộng sự cố.

Chạy: python -m src.scenarios [--mode smoke|full]
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from src.algorithms.td_astar import td_astar
from src.config import Config, load_config
from src.graph_data import RoadGraph
from src.speed_model import Incident, TravelTimeModel, base_model, load_vbase, real_V
from src.utils import derive_seed, load_json, save_json, setup_logging

log = logging.getLogger("scenarios")
SPLITS = ("train", "val", "test")


def _corridor_edges(graph: RoadGraph, o: int, d: int, width_m: float) -> np.ndarray:
    """Các cạnh có trung điểm cách đoạn thẳng O–D không quá width_m."""
    ax, ay = graph.node_x[o], graph.node_y[o]
    bx, by = graph.node_x[d], graph.node_y[d]
    mx = (graph.node_x[graph.edge_from] + graph.node_x[graph.edge_to]) / 2
    my = (graph.node_y[graph.edge_from] + graph.node_y[graph.edge_to]) / 2
    vx, vy = bx - ax, by - ay
    tproj = np.clip(((mx - ax) * vx + (my - ay) * vy) / max(vx * vx + vy * vy, 1e-9), 0, 1)
    dist = np.hypot(mx - (ax + tproj * vx), my - (ay + tproj * vy))
    return np.flatnonzero(dist <= width_m)


def sample_od_pairs(graph: RoadGraph, n: int, min_dist: float, rng: np.random.Generator,
                    max_tries: int) -> list[tuple[int, int]]:
    """Lấy n cặp O–D khác nhau, khoảng cách chim bay ≥ min_dist (đồ thị liên thông mạnh nên luôn có đường)."""
    pairs: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"Không lấy đủ {n} cặp O–D sau {max_tries} lần thử")
        o, d = (int(x) for x in rng.integers(0, graph.n_nodes, 2))
        if o == d or (o, d) in pairs or graph.straight_dist(o, d) < min_dist:
            continue
        pairs.add((o, d))
        out.append((o, d))
    return out


def generate(cfg: Config, graph: RoadGraph, V_base: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """Sinh toàn bộ truy vấn và chia theo cặp O–D."""
    sc = cfg["scenarios"]
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(derive_seed(seed, "scenarios", cfg.mode))
    n = int(sc["n_queries"])
    pairs = sample_od_pairs(graph, n, float(sc["min_od_dist_m"]), rng, n * int(sc["max_tries_factor"]))
    perm = rng.permutation(n)
    n_tr = int(round(sc["split"][0] * n))
    n_va = int(round(sc["split"][1] * n))
    split_of = {int(i): ("train" if k < n_tr else "val" if k < n_tr + n_va else "test") for k, i in enumerate(perm)}

    hw = np.asarray(sc["hour_weights"], dtype=float)
    hw /= hw.sum()
    forecast = base_model(graph, V_base, cfg)
    bins = np.asarray(sc["distance_bins_m"], dtype=float)
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    train_incident_edges: set[int] = set()
    # xử lý train trước để tập test tránh được vị trí sự cố đã thấy ở train
    order = sorted(range(n), key=lambda i: SPLITS.index(split_of[i]))
    for i in order:
        split = split_of[i]
        o, d = pairs[i]
        hour = int(rng.choice(24, p=hw))
        depart = float(hour * 60 + rng.uniform(0, 60))
        sigma = float(rng.choice(sc["sigma_levels"]))
        plan = td_astar(graph, o, d, depart, forecast)
        n_inc = int(rng.choice(sc["n_incidents_choices"], p=sc["n_incidents_probs"]))
        forbidden = train_incident_edges if split == "test" else set()
        route = [e for e in plan.edges if e not in forbidden]
        corridor = [int(e) for e in _corridor_edges(graph, o, d, sc["incident_corridor_m"]) if int(e) not in forbidden]
        ic = cfg["speed"]["incidents"]
        off_lo, off_hi = sc["incident_start_offset_min"]
        sets: dict[str, list[dict[str, float]]] = {}
        chosen_all: list[int] = []
        # cùng số sự cố, cùng thời điểm/mức độ; chỉ khác VỊ TRÍ: tập chính ngẫu nhiên, tập stress ưu tiên trên tuyến
        timing = [(depart + float(rng.uniform(off_lo, off_hi)), float(rng.uniform(*ic["duration_range_min"])),
                   float(rng.uniform(*ic["severity_range"]))) for _ in range(n_inc)]
        for key, p_route in (("incidents", sc["incident_on_route_prob"]), ("incidents_stress", sc["stress_on_route_prob"])):
            chosen: list[int] = []
            for _ in range(n_inc):
                pool = route if (route and rng.random() < p_route) else corridor
                pool = [e for e in pool if e not in chosen]
                if pool:
                    chosen.append(int(rng.choice(pool)))
            sets[key] = [Incident(e, *timing[j]).to_dict() for j, e in enumerate(chosen)]
            chosen_all += chosen
        chosen = chosen_all
        if split == "train":
            train_incident_edges.update(chosen)
        dist = graph.straight_dist(o, d)
        out[split].append({
            "id": f"{split}_{len(out[split]):04d}", "split": split, "origin": o, "dest": d,
            "depart_min": round(depart, 3), "hour": hour, "sigma": sigma,
            "noise_seed": derive_seed(seed, "noise", i, cfg.mode), **sets,
            "od_dist_m": round(dist, 1), "distance_bin": int(np.clip(np.searchsorted(bins, dist, side="right") - 1, 0, len(bins) - 2)),
            "ref_hops": len(plan.edges), "ref_time_base_min": round(plan.cost, 4),
        })
    return out


def build_real_model(graph: RoadGraph, V_base: np.ndarray, q: dict[str, Any], cfg: Config,
                     sigma: float | None = None, with_incidents: bool = True,
                     incident_key: str = "incidents") -> TravelTimeModel:
    """Mô hình thời gian THỰC của truy vấn q (tái lập theo noise_seed).

    incident_key: "incidents" (tập chính, vị trí ngẫu nhiên) hoặc "incidents_stress" (ưu tiên trên tuyến dự kiến).
    """
    s = q["sigma"] if sigma is None else sigma
    V = real_V(graph, V_base, s, int(q["noise_seed"]), cfg)
    incs = [Incident(**i) for i in q.get(incident_key, [])] if with_incidents else []
    return base_model(graph, V, cfg, incs)


def load_split(cfg: Config, split: str) -> list[dict[str, Any]]:
    path = cfg.path("eval_dir") / f"{split}_set.json"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path} — chạy `python -m src.scenarios --mode {cfg.mode}`")
    return load_json(path)["queries"]


def run(cfg: Config) -> dict[str, Any]:
    graph = RoadGraph.from_config(cfg)
    V_base = load_vbase(cfg)
    sets = generate(cfg, graph, V_base)
    eval_dir = cfg.path("eval_dir")
    keys = ("incidents", "incidents_stress")
    train_inc = {i["edge"] for q in sets["train"] for k in keys for i in q[k]}
    for split, qs in sets.items():
        save_json({"mode": cfg.mode, "seed": cfg["project"]["seed"], "n": len(qs), "queries": qs},
                  eval_dir / f"{split}_set.json")
    pairs = {s: {(q["origin"], q["dest"]) for q in qs} for s, qs in sets.items()}
    test_inc = {i["edge"] for q in sets["test"] for k in keys for i in q[k]}
    summary = {
        "n": {s: len(q) for s, q in sets.items()},
        "od_overlap_train_test": len(pairs["train"] & pairs["test"]),
        "od_overlap_train_val": len(pairs["train"] & pairs["val"]),
        "test_incident_edges_seen_in_train": len(test_inc & train_inc),
        "mean_od_dist_m": {s: float(np.mean([q["od_dist_m"] for q in qs])) for s, qs in sets.items()},
        "mean_incidents": {s: float(np.mean([len(q["incidents"]) for q in qs])) for s, qs in sets.items()},
        "hour_hist_all": np.bincount([q["hour"] for qs in sets.values() for q in qs], minlength=24).tolist(),
    }
    save_json(summary, cfg.path("results_dir") / "scenarios_summary.json")
    log.info("Truy vấn: %s; trùng O–D train/test=%d; cạnh sự cố test đã thấy ở train=%d; KC TB test=%.0f m",
             summary["n"], summary["od_overlap_train_test"], summary["test_incident_edges_seen_in_train"],
             summary["mean_od_dist_m"]["test"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg)


if __name__ == "__main__":
    main()
