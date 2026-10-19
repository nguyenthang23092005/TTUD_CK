"""Dijkstra tĩnh trên CSR (heapq), hỗ trợ cấm cạnh/nút (cho Yen) và chạy một-nguồn-mọi-đích."""
from __future__ import annotations

import heapq
import time
from typing import Sequence

import numpy as np

from src.algorithms.common import SearchResult, nodes_from_edges, reconstruct
from src.graph_data import RoadGraph


def dijkstra(graph: RoadGraph, source: int, target: int, weights: Sequence[float] | np.ndarray,
             banned_edges: set[int] | None = None, banned_nodes: set[int] | None = None,
             heuristic: Sequence[float] | None = None) -> SearchResult:
    """Dijkstra (hoặc A* nếu có `heuristic` theo nút) với trọng số tĩnh không âm.

    Args:
        weights: trọng số theo cạnh (độ dài E).
        banned_edges / banned_nodes: các cạnh/nút không được dùng.
        heuristic: h(n) admissible & consistent; None -> Dijkstra thuần.
    """
    t_start = time.perf_counter()
    w = weights.tolist() if isinstance(weights, np.ndarray) else weights
    adj = graph.adj
    be = banned_edges or set()
    bn = banned_nodes or set()
    h = heuristic
    dist: dict[int, float] = {source: 0.0}
    pred: dict[int, int] = {}
    closed: set[int] = set()
    heap: list[tuple[float, float, int]] = [((h[source] if h is not None else 0.0), 0.0, source)]
    expanded = 0
    while heap:
        _, d, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == target:
            edges, _ = reconstruct(pred, graph.edge_from, source, target)
            return SearchResult(True, nodes_from_edges(source, edges, graph.edge_to_list), edges, d, None,
                                expanded, time.perf_counter() - t_start)
        for e, v in adj[u]:
            if v in closed or e in be or v in bn:
                continue
            nd = d + w[e]
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                pred[v] = e
                heapq.heappush(heap, (nd + (h[v] if h is not None else 0.0), nd, v))
    return SearchResult(False, expanded=expanded, runtime_s=time.perf_counter() - t_start)


def dijkstra_all(graph: RoadGraph, source: int, weights: Sequence[float] | np.ndarray,
                 reverse: bool = False) -> np.ndarray:
    """Khoảng cách từ `source` tới mọi nút (reverse=True: từ mọi nút TỚI `source`)."""
    w = weights.tolist() if isinstance(weights, np.ndarray) else weights
    adj = graph.radj if reverse else graph.adj
    dist = np.full(graph.n_nodes, np.inf)
    dist[source] = 0.0
    done = np.zeros(graph.n_nodes, dtype=bool)
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for e, v in adj[u]:
            nd = d + w[e]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def shortest_by_length(graph: RoadGraph, source: int, target: int) -> SearchResult:
    """Tuyến ngắn nhất theo chiều dài (m)."""
    return dijkstra(graph, source, target, graph.length_list)


def freeflow_weights(graph: RoadGraph) -> list[float]:
    """Thời gian tự do L / v_free (phút)."""
    return (graph.length / (graph.v_free * 1000.0 / 60.0)).tolist()
