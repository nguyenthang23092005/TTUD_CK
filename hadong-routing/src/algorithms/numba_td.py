"""TD-A* biên dịch JIT bằng Numba (mục C4) — cùng thuật toán với `td_astar.td_search`, chạy trên mảng CSR.

Hỗ trợ mô hình vận tốc dạng ma trận (E, 24) không có sự cố (sự cố dùng bảng FIFO hoá → dùng bản Python).
Contraction Hierarchies (CH/TCH) KHÔNG được cài: ở quy mô ~3.500 nút, TD-A* Numba đã dưới 1 ms/truy vấn; CH chỉ
có lợi khi mở rộng ra toàn thành phố (hàng trăm nghìn nút) và bản phụ thuộc thời gian (TCH) phức tạp hơn nhiều.
"""
from __future__ import annotations

import heapq
import time

import numba as nb
import numpy as np

from src.algorithms.common import SearchResult, nodes_from_edges
from src.graph_data import RoadGraph
from src.speed_model import TravelTimeModel


@nb.njit(cache=True)
def _td_astar_nb(out_ptr, out_edges, edge_to, length, V_mpm, h, source, target, t0):  # pragma: no cover - JIT
    n = out_ptr.shape[0] - 1
    arr = np.full(n, np.inf)
    closed = np.zeros(n, np.bool_)
    pred = np.full(n, -1, np.int64)
    arr[source] = t0
    heap = [(t0 + h[source], t0, np.int64(source))]
    expanded = 0
    while len(heap) > 0:
        item = heapq.heappop(heap)
        t = item[1]
        u = item[2]
        if closed[u]:
            continue
        closed[u] = True
        expanded += 1
        if u == target:
            break
        tm = t % 1440.0
        hh = int(tm // 60.0)
        fr = (tm - 60.0 * hh) / 60.0
        h1 = hh + 1 if hh < 23 else 0
        for k in range(out_ptr[u], out_ptr[u + 1]):
            e = out_edges[k]
            v = edge_to[e]
            if closed[v]:
                continue
            sp = V_mpm[e, hh] * (1.0 - fr) + V_mpm[e, h1] * fr
            ta = t + length[e] / sp
            if ta < arr[v]:
                arr[v] = ta
                pred[v] = e
                heapq.heappush(heap, (ta + h[v], ta, np.int64(v)))
    return arr[target], pred, expanded


def td_astar_numba(graph: RoadGraph, source: int, target: int, t0: float, model: TravelTimeModel,
                   heuristic: np.ndarray | None = None, safety: float = 0.995) -> SearchResult:
    """TD-A* Numba; heuristic mặc định = chim bay / v_max."""
    if model.incidents:
        raise ValueError("td_astar_numba không hỗ trợ sự cố — dùng td_astar (Python)")
    t_start = time.perf_counter()
    h = heuristic if heuristic is not None else graph.dist_to(target) * (safety / model.v_max_mpm)
    cost, pred, expanded = _td_astar_nb(graph.out_ptr, graph.out_edges, graph.edge_to, graph.length,
                                        model.V_mpm, np.asarray(h, dtype=np.float64), np.int64(source),
                                        np.int64(target), float(t0))
    if not np.isfinite(cost):
        return SearchResult(False, expanded=int(expanded), runtime_s=time.perf_counter() - t_start)
    edges: list[int] = []
    v = target
    while v != source:
        e = int(pred[v])
        edges.append(e)
        v = int(graph.edge_from[e])
    edges.reverse()
    return SearchResult(True, nodes_from_edges(source, edges, graph.edge_to_list), edges, float(cost - t0),
                        float(cost), int(expanded), time.perf_counter() - t_start)


def warmup(graph: RoadGraph, model: TravelTimeModel) -> None:
    """Biên dịch JIT trước (không tính vào thời gian đo)."""
    td_astar_numba(graph, 0, int(graph.edge_to[0]), 0.0, model)
