"""Mô phỏng nhiều xe cùng được dẫn đường (mục E3): khi mọi xe làm theo cùng một chỉ dẫn, ùn tắc dịch chuyển.

Mô hình tải: mỗi tác tử đại diện `vehicles_per_agent` xe. Lưu lượng dẫn đường trên cạnh e trong khung `slot_min`
phút được quy ra xe/giờ và làm chậm cạnh theo hàm BPR (Bureau of Public Roads):
    τ_load(e,t) = τ_base(e,t) · (1 + α·(q_e(slot)/cap_e)^β)
(lưu lượng nền đã nằm trong V_base; đây là phần tăng thêm do xe được dẫn đường). Hai chiến lược:
  * naive       — mọi xe chạy TD-A* trên V_base, không biết xe khác (chỉ dẫn "ích kỷ" giống nhau);
  * coordinated — lập kế hoạch tuần tự theo giờ xuất phát, mỗi xe thấy tải của các xe đã lập trước (phân bổ tham lam).
Thời gian thực của cả hai được tính bằng điểm bất động: tải ↔ lịch trình trên tuyến cố định.

Chạy: python -m src.evaluation.multiagent [--mode smoke|full]
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.algorithms.td_astar import td_astar
from src.config import Config, load_config
from src.graph_data import RoadGraph
from src.scenarios import sample_od_pairs
from src.speed_model import TravelTimeModel, base_model, load_vbase
from src.utils import derive_seed, setup_logging

log = logging.getLogger("multiagent")


class LoadModel:
    """Mô hình thời gian có tải dẫn đường (BPR). v_max giữ nguyên ⇒ heuristic TD-A* vẫn admissible."""

    def __init__(self, base: TravelTimeModel, load: np.ndarray, cap_vph: np.ndarray, mc: dict[str, Any]) -> None:
        self.base = base
        self.load = load                      # (E, n_slot) số tác tử vào cạnh trong khung
        self.cap = cap_vph
        self.slot = float(mc["slot_min"])
        self.scale = float(mc["vehicles_per_agent"]) * 60.0 / self.slot   # tác tử/khung → xe/giờ
        self.alpha, self.beta = float(mc["bpr_alpha"]), float(mc["bpr_beta"])
        self.v_max_mpm = base.v_max_mpm

    def factor(self, e: int, t: float) -> float:
        k = int(t // self.slot)
        if k < 0 or k >= self.load.shape[1]:
            return 1.0
        q = self.load[e, k] * self.scale
        return 1.0 + self.alpha * (q / self.cap[e]) ** self.beta if q > 0 else 1.0

    def travel_time(self, e: int, t: float) -> float:
        return self.base.travel_time(e, t) * self.factor(e, t)


def _schedule(model: Any, edges: list[int], t0: float) -> tuple[list[float], float]:
    times, t = [], t0
    for e in edges:
        times.append(t)
        t += model.travel_time(e, t)
    return times, t - t0


def _add(load: np.ndarray, edges: list[int], times: list[float], slot: float, sign: float = 1.0) -> None:
    for e, t in zip(edges, times):
        k = int(t // slot)
        if 0 <= k < load.shape[1]:
            load[e, k] += sign


def evaluate_paths(base: TravelTimeModel, paths: list[list[int]], departs: list[float], cap: np.ndarray,
                   mc: dict[str, Any], n_slot: int) -> tuple[np.ndarray, np.ndarray]:
    """Thời gian đi với tải nhất quán (lặp điểm bất động trên tuyến cố định)."""
    load = np.zeros((len(cap), n_slot))
    scheds = [_schedule(base, p, d)[0] for p, d in zip(paths, departs)]
    tts = np.zeros(len(paths))
    for _ in range(int(mc["fixed_point_iters"])):
        load[:] = 0
        for p, s in zip(paths, scheds):
            _add(load, p, s, mc["slot_min"])
        lm = LoadModel(base, load, cap, mc)
        out = [_schedule(lm, p, d) for p, d in zip(paths, departs)]
        scheds = [o[0] for o in out]
        tts = np.array([o[1] for o in out])
    return tts, load


def run(cfg: Config) -> pd.DataFrame:
    mc = cfg["multiagent"]
    graph = RoadGraph.from_config(cfg)
    V = load_vbase(cfg)
    base = base_model(graph, V, cfg)
    cap = np.array([mc["capacity_vph"][graph.group_names[g]] for g in graph.group_id], dtype=float)
    n_slot = int(1440 * 2 // mc["slot_min"])
    rows = []
    lo, hi = mc["depart_window_min"]
    for n in mc["n_agents_list"]:
        rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "multiagent", int(n)))
        pairs = sample_od_pairs(graph, int(n), float(mc["min_od_dist_m"]), rng, 1000 * int(n))
        departs = sorted(float(x) for x in rng.uniform(lo, hi, size=int(n)))
        t0 = time.perf_counter()
        naive = [td_astar(graph, o, d, dep, base).edges for (o, d), dep in zip(pairs, departs)]
        free_tt = np.array([base.path_travel_time(p, dep) for p, dep in zip(naive, departs)])
        tt_naive, load_n = evaluate_paths(base, naive, departs, cap, mc, n_slot)
        load = np.zeros((graph.n_edges, n_slot))
        coord = []
        for (o, d), dep in zip(pairs, departs):
            lm = LoadModel(base, load, cap, mc)
            p = td_astar(graph, o, d, dep, lm).edges
            coord.append(p)
            _add(load, p, _schedule(lm, p, dep)[0], mc["slot_min"])
        tt_coord, load_c = evaluate_paths(base, coord, departs, cap, mc, n_slot)
        scale = float(mc["vehicles_per_agent"]) * 60.0 / mc["slot_min"]
        for name, tt, ld in (("no_interaction", free_tt, None), ("naive", tt_naive, load_n), ("coordinated", tt_coord, load_c)):
            vc = (ld * scale / cap[:, None]).max() if ld is not None else 0.0
            rows.append({"n_agents": int(n), "n_vehicles": int(n) * int(mc["vehicles_per_agent"]), "policy": name,
                         "tt_mean": float(tt.mean()), "tt_p90": float(np.quantile(tt, 0.9)),
                         "max_vc": float(vc), "edges_used": int(len({e for p in (naive if name != "coordinated" else coord) for e in p}))})
        log.info("N=%d tác tử (%d xe): TB không tương tác %.2f | naive %.2f | phối hợp %.2f phút (%.1f s)", n,
                 n * mc["vehicles_per_agent"], free_tt.mean(), tt_naive.mean(), tt_coord.mean(), time.perf_counter() - t0)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.path("results_dir") / "multiagent.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg)


if __name__ == "__main__":
    main()
