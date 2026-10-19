"""Yen (1971): k tuyến ngắn nhất không lặp với trọng số tĩnh."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.algorithms.common import SearchResult
from src.algorithms.dijkstra import dijkstra
from src.graph_data import RoadGraph


def yen_k_shortest(graph: RoadGraph, source: int, target: int, weights: Sequence[float] | np.ndarray,
                   k: int = 3) -> list[SearchResult]:
    """Trả về tối đa k tuyến (tăng dần theo chi phí), mỗi tuyến là một SearchResult."""
    w = weights.tolist() if isinstance(weights, np.ndarray) else list(weights)
    first = dijkstra(graph, source, target, w)
    if not first.found:
        return []
    A: list[SearchResult] = [first]
    B: list[tuple[float, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = {tuple(first.edges)}
    for _ in range(1, k):
        prev = A[-1]
        for i in range(len(prev.edges)):
            spur = prev.nodes[i]
            root_edges = prev.edges[:i]
            banned_edges = {p.edges[i] for p in A if len(p.edges) > i and p.edges[:i] == root_edges}
            banned_nodes = set(prev.nodes[:i])
            r = dijkstra(graph, spur, target, w, banned_edges=banned_edges, banned_nodes=banned_nodes)
            if not r.found:
                continue
            cand = tuple(root_edges + r.edges)
            if cand in seen:
                continue
            seen.add(cand)
            B.append((sum(w[e] for e in cand), cand))
        if not B:
            break
        B.sort(key=lambda x: x[0])
        cost, edges = B.pop(0)
        nodes = [source] + [graph.edge_to_list[e] for e in edges]
        A.append(SearchResult(True, nodes, list(edges), cost))
    return A
