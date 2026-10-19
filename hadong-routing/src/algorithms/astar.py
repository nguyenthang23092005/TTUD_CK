"""A* tĩnh với heuristic khoảng cách chim bay (Haversine/chiếu phẳng) trên CSR."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.algorithms.common import SearchResult
from src.algorithms.dijkstra import dijkstra, freeflow_weights
from src.graph_data import RoadGraph


def straight_line_heuristic(graph: RoadGraph, target: int, scale: float = 1.0, safety: float = 0.995) -> list[float]:
    """h(n) = khoảng_cách_chim_bay(n, target) * scale * safety (scale=1 cho mét, 1/v_max cho phút)."""
    return (graph.dist_to(target) * scale * safety).tolist()


def astar(graph: RoadGraph, source: int, target: int, weights: Sequence[float] | np.ndarray,
          heuristic: Sequence[float]) -> SearchResult:
    """A* tổng quát với heuristic theo nút đã tính sẵn."""
    return dijkstra(graph, source, target, weights, heuristic=heuristic)


def astar_length(graph: RoadGraph, source: int, target: int, safety: float = 0.995) -> SearchResult:
    """A* theo chiều dài; heuristic chim bay là admissible vì chiều dài cạnh ≥ khoảng cách hai đầu."""
    return astar(graph, source, target, graph.length_list, straight_line_heuristic(graph, target, 1.0, safety))


def astar_freeflow(graph: RoadGraph, source: int, target: int, safety: float = 0.995) -> SearchResult:
    """A* theo thời gian tự do; heuristic = chim bay / max(v_free)."""
    v_max_mpm = float(graph.v_free.max()) * 1000.0 / 60.0
    return astar(graph, source, target, freeflow_weights(graph),
                 straight_line_heuristic(graph, target, 1.0 / v_max_mpm, safety))
