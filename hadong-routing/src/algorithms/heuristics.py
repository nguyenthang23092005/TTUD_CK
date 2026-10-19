"""Heuristic chặt hơn cho TD-A* (mục C2).

* Cận dưới chính xác (LB): τ_min(e) = L_e / max_t v_e(t) là cận dưới của τ_e(t) với mọi t (sự cố chỉ làm chậm),
  h_LB(n) = khoảng cách ngắn nhất từ n tới đích trên đồ thị trọng số τ_min (một lần Dijkstra ngược).
  h_LB admissible và consistent: h(u) ≤ τ_min(u,v) + h(v) ≤ τ(u,v,t) + h(v). Đây là heuristic tĩnh chặt nhất
  có thể có từ cận dưới theo cạnh, nhưng tốn một lần Dijkstra toàn đồ thị cho mỗi đích.
* ALT (Goldberg & Harrelson 2005): chọn L điểm mốc, tiền tính khoảng cách tới/từ mốc trên đồ thị τ_min một lần;
  h(n) = max_L max(d(n,L) − d(t,L), d(L,t) − d(L,n)) — admissible theo bất đẳng thức tam giác, truy vấn O(L).
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from src.algorithms.dijkstra import dijkstra_all
from src.graph_data import RoadGraph


def min_travel_time(length_m: np.ndarray, V_kmh: np.ndarray) -> np.ndarray:
    """τ_min(e) (phút) = L_e / max_h V(e,h)."""
    return length_m / (V_kmh.max(axis=1) * 1000.0 / 60.0)


def lb_heuristic(graph: RoadGraph, target: int, tau_min: np.ndarray, safety: float = 0.999) -> tuple[list[float], float]:
    """h_LB cho mọi nút và thời gian tiền xử lý (giây)."""
    t0 = time.perf_counter()
    d = dijkstra_all(graph, target, tau_min, reverse=True)
    return (d * safety).tolist(), time.perf_counter() - t0


class ALT:
    """Điểm mốc chọn bằng farthest-point trên đồ thị τ_min; khoảng cách tiền tính hai chiều."""

    def __init__(self, graph: RoadGraph, tau_min: np.ndarray, n_landmarks: int = 8, seed: int = 0,
                 safety: float = 0.999) -> None:
        t0 = time.perf_counter()
        self.graph = graph
        self.safety = safety
        rng = np.random.default_rng(seed)
        first = int(rng.integers(graph.n_nodes))
        lms: list[int] = []
        far = dijkstra_all(graph, first, tau_min)
        for _ in range(n_landmarks):
            nxt = int(np.argmax(np.where(np.isfinite(far), far, -1)))
            lms.append(nxt)
            d = dijkstra_all(graph, nxt, tau_min)
            far = d if len(lms) == 1 else np.minimum(far, d)
        self.landmarks = lms
        self.d_from = np.stack([dijkstra_all(graph, L, tau_min) for L in lms])            # d(L, ·)
        self.d_to = np.stack([dijkstra_all(graph, L, tau_min, reverse=True) for L in lms])  # d(·, L)
        self.preprocess_s = time.perf_counter() - t0

    def heuristic(self, target: int) -> list[float]:
        """Vector h(n) cho mọi nút với đích `target` (vector hoá numpy)."""
        a = self.d_to - self.d_to[:, [target]]      # d(n,L) − d(t,L)
        b = self.d_from[:, [target]] - self.d_from  # d(L,t) − d(L,n)
        h = np.maximum(np.maximum(a, b).max(axis=0), 0.0)
        return (h * self.safety).tolist()


def heuristic_quality(h: Sequence[float], exact: Sequence[float]) -> float:
    """Tỉ lệ trung bình h/exact (1 = hoàn hảo) trên các nút có exact > 0."""
    h, exact = np.asarray(h), np.asarray(exact)
    m = exact > 1e-9
    return float(np.mean(h[m] / exact[m]))
