"""TD-Dijkstra và TD-A* (time-dependent) trên mạng FIFO.

Nhãn của nút = thời điểm đến sớm nhất. Với FIFO, giải thuật gắn nhãn cố định (label-setting) cho kết quả tối ưu
(Dreyfus 1969; Kaufman & Smith 1993). TD-A* dùng h(n) = khoảng_cách_chim_bay(n, đích) / v_max — admissible và
consistent vì τ(e,t) ≥ L_e / v_max ≥ chim_bay(u,v) / v_max.
"""
from __future__ import annotations

import heapq
import time
from typing import Sequence

from src.algorithms.common import SearchResult, TimeModel, nodes_from_edges, reconstruct
from src.graph_data import RoadGraph


def td_search(graph: RoadGraph, source: int, target: int, t0: float, model: TimeModel,
              heuristic: Sequence[float] | None = None) -> SearchResult:
    """Tìm tuyến đến sớm nhất xuất phát lúc t0 (phút). heuristic=None -> TD-Dijkstra."""
    t_start = time.perf_counter()
    adj = graph.adj
    tt = model.travel_time
    h = heuristic
    arr: dict[int, float] = {source: t0}
    pred: dict[int, int] = {}
    closed: set[int] = set()
    heap: list[tuple[float, float, int]] = [((h[source] if h is not None else 0.0) + t0, t0, source)]
    expanded = 0
    while heap:
        _, t, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == target:
            edges, _ = reconstruct(pred, graph.edge_from, source, target)
            return SearchResult(True, nodes_from_edges(source, edges, graph.edge_to_list), edges, t - t0, t,
                                expanded, time.perf_counter() - t_start)
        for e, v in adj[u]:
            if v in closed:
                continue
            ta = t + tt(e, t)
            if ta < arr.get(v, float("inf")):
                arr[v] = ta
                pred[v] = e
                heapq.heappush(heap, (ta + (h[v] if h is not None else 0.0), ta, v))
    return SearchResult(False, expanded=expanded, runtime_s=time.perf_counter() - t_start)


def td_dijkstra(graph: RoadGraph, source: int, target: int, t0: float, model: TimeModel) -> SearchResult:
    return td_search(graph, source, target, t0, model, None)


def td_heuristic(graph: RoadGraph, target: int, v_max_mpm: float, safety: float = 0.995) -> list[float]:
    """h(n) = chim_bay(n, target) / v_max (phút)."""
    return (graph.dist_to(target) * (safety / v_max_mpm)).tolist()


def td_astar(graph: RoadGraph, source: int, target: int, t0: float, model: TimeModel,
             heuristic: Sequence[float] | None = None, safety: float = 0.995) -> SearchResult:
    """TD-A*; nếu không truyền heuristic thì tính từ model.v_max_mpm."""
    if heuristic is None:
        heuristic = td_heuristic(graph, target, model.v_max_mpm, safety)
    return td_search(graph, source, target, t0, model, heuristic)
