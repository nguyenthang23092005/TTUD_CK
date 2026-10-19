"""Định tuyến tính đến rủi ro (mục D2): chọn tuyến cực tiểu phân vị q (mặc định P90) của thời gian đi.

Phân phối thời gian đi được ước lượng bằng một "ngân hàng kịch bản": M trường nhiễu (cùng mô hình nhiễu tương quan
của môi trường, seed độc lập với seed của truy vấn) ⇒ M ma trận vận tốc. Ứng viên = k tuyến Yen theo thời gian tại
giờ xuất phát + tuyến TD-A* trên dự báo. Thời gian của một tuyến trên M kịch bản được tính vector hoá (numpy).
Bài toán "đường đi cực tiểu phân vị" không có cấu trúc con tối ưu (không cộng tính) nên dùng tập ứng viên.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from src.algorithms.replan import TripResult, simulate
from src.algorithms.td_astar import td_astar
from src.algorithms.yen import yen_k_shortest
from src.graph_data import RoadGraph
from src.speed_model import KMH_TO_MPM, NoiseModel, TravelTimeModel


class ScenarioBank:
    """M kịch bản vận tốc (m/phút) cho mỗi mức σ."""

    def __init__(self, V_base: np.ndarray, length: np.ndarray, noise: NoiseModel, sigmas: Sequence[float],
                 n_samples: int, seed: int) -> None:
        self.length = length
        self.banks: dict[float, np.ndarray] = {}
        for j, s in enumerate(sigmas):
            fields = [np.exp(noise.log_eps(float(s), seed + 1000 * j + m)) for m in range(n_samples)]
            self.banks[float(s)] = (V_base[None] * np.stack(fields) * KMH_TO_MPM).astype(np.float32)

    def bank(self, sigma: float) -> np.ndarray:
        key = min(self.banks, key=lambda s: abs(s - sigma))
        return self.banks[key]


def path_times(V_mpm: np.ndarray, length: np.ndarray, edges: Sequence[int], t0: float) -> np.ndarray:
    """Thời gian (phút) của tuyến trên mọi kịch bản: V_mpm có dạng (M, E, 24)."""
    M = V_mpm.shape[0]
    idx = np.arange(M)
    t = np.full(M, t0, dtype=np.float64)
    for e in edges:
        tm = np.mod(t, 1440.0)
        h = (tm // 60.0).astype(np.int64)
        f = (tm - 60.0 * h) / 60.0
        v = V_mpm[idx, e, h] * (1.0 - f) + V_mpm[idx, e, (h + 1) % 24] * f
        t = t + length[e] / v
    return t - t0


def risk_route(graph: RoadGraph, source: int, target: int, t0: float, forecast: TravelTimeModel,
               real: TravelTimeModel, V_bank: np.ndarray, k: int = 6, quantile: float = 0.9) -> TripResult:
    """Chọn tuyến có phân vị `quantile` nhỏ nhất trong tập ứng viên, rồi đi trong V_real."""
    t_start = time.perf_counter()
    snap = graph.length / np.maximum(forecast.base_speed_mpm(np.arange(graph.n_edges), t0), 1e-6)
    cands = {tuple(p.edges) for p in yen_k_shortest(graph, source, target, snap, k)}
    po = td_astar(graph, source, target, t0, forecast)
    cands.add(tuple(po.edges))
    best, best_q, best_mean = None, np.inf, np.nan
    for c in cands:
        tt = path_times(V_bank, graph.length, c, t0)
        qv = float(np.quantile(tt, quantile))
        if qv < best_q:
            best, best_q, best_mean = list(c), qv, float(tt.mean())
    res = TripResult("risk_p90", True, best, [source] + [graph.edge_to_list[e] for e in best],
                     simulate(graph, real, best, t0), best_mean, graph.path_length(best), po.expanded,
                     time.perf_counter() - t_start)
    res.extra = {"n_candidates": len(cands), "planned_quantile": best_q, "same_as_plan_once": best == po.edges}
    return res
