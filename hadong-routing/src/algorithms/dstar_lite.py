"""D* Lite (Koenig & Likhachev 2002) — tái lập kế hoạch tăng dần (mục C1).

D* Lite giả định chi phí cạnh TĨNH và chỉ thay đổi khi có quan sát mới. Mạng của ta phụ thuộc thời gian, nên dùng
xấp xỉ "snapshot": c(e) = τ_obs(e, t_snapshot) cố định cho tới khi xe đi thêm `refresh_min` phút thì dựng lại toàn bộ.
Giữa hai lần dựng lại, chỉ các cạnh có quan sát mới bị cập nhật và D* Lite sửa lời giải cục bộ thay vì tìm lại từ đầu.
Đánh đổi: nhanh hơn TD-A* lặp, nhưng bỏ qua biến thiên vận tốc trong tương lai gần (mất tính tối ưu theo thời gian).
"""
from __future__ import annotations

import heapq
import math
import time
from typing import Callable

import numpy as np

from src.algorithms.replan import ObservedModel, TripResult
from src.graph_data import RoadGraph
from src.speed_model import TravelTimeModel

INF = float("inf")


class DStarLite:
    """D* Lite trên đồ thị có hướng với chi phí cạnh `cost[e]` (danh sách, có thể cập nhật)."""

    def __init__(self, graph: RoadGraph, start: int, goal: int, cost: list[float], h_scale: float) -> None:
        self.g_ = graph
        self.cost = cost
        self.start, self.goal = start, goal
        self.h_scale = h_scale  # h(a,b) = chim_bay(a,b) * h_scale (≤ chi phí thật vì h_scale ≤ 1/v_max)
        self.g: dict[int, float] = {}
        self.rhs: dict[int, float] = {goal: 0.0}
        self.km = 0.0
        self.U: list[tuple[float, float, int]] = []
        self.in_U: dict[int, tuple[float, float]] = {}
        self.expanded = 0
        self._push(goal, self._key(goal))

    # -- tiện ích -----------------------------------------------------------
    def _h(self, a: int, b: int) -> float:
        g = self.g_
        return math.hypot(g.node_x[a] - g.node_x[b], g.node_y[a] - g.node_y[b]) * self.h_scale

    def _key(self, s: int) -> tuple[float, float]:
        m = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (m + self._h(self.start, s) + self.km, m)

    def _push(self, s: int, k: tuple[float, float]) -> None:
        self.in_U[s] = k
        heapq.heappush(self.U, (k[0], k[1], s))

    def _top(self) -> tuple[tuple[float, float], int] | None:
        while self.U:
            k1, k2, s = self.U[0]
            if self.in_U.get(s) == (k1, k2):
                return (k1, k2), s
            heapq.heappop(self.U)  # bản ghi cũ (lazy deletion)
        return None

    def update_vertex(self, u: int) -> None:
        if u != self.goal:
            best = INF
            for e, v in self.g_.adj[u]:
                c = self.cost[e] + self.g.get(v, INF)
                if c < best:
                    best = c
            self.rhs[u] = best
        self.in_U.pop(u, None)
        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self._push(u, self._key(u))

    def compute_shortest_path(self) -> None:
        while True:
            top = self._top()
            if top is None:
                break
            k_old, u = top
            k_start = self._key(self.start)
            if not (k_old < k_start or self.rhs.get(self.start, INF) != self.g.get(self.start, INF)):
                break
            self.expanded += 1
            k_new = self._key(u)
            if k_old < k_new:
                self._push(u, k_new)
            elif self.g.get(u, INF) > self.rhs.get(u, INF):
                self.g[u] = self.rhs[u]
                self.in_U.pop(u, None)
                for _, p in self.g_.radj[u]:
                    self.update_vertex(p)
            else:
                self.g[u] = INF
                for _, p in self.g_.radj[u]:
                    self.update_vertex(p)
                self.update_vertex(u)

    def next_edge(self) -> int | None:
        """Cạnh ra tốt nhất từ start theo g hiện tại."""
        best, be = INF, None
        for e, v in self.g_.adj[self.start]:
            c = self.cost[e] + self.g.get(v, INF)
            if c < best:
                best, be = c, e
        return be

    def move_to(self, s: int) -> None:
        self.km += self._h(self.start, s)
        self.start = s

    def change_cost(self, e: int, new_cost: float) -> None:
        if new_cost != self.cost[e]:
            self.cost[e] = new_cost
            self.update_vertex(int(self.g_.edge_from[e]))

    def path_cost(self) -> float:
        return self.g.get(self.start, INF)


def snapshot_costs(model: Callable[[int, float], float], n_edges: int, t: float) -> list[float]:
    return [model(e, t) for e in range(n_edges)]


def replan_dstar(graph: RoadGraph, source: int, target: int, t0: float, forecast: TravelTimeModel,
                 real: TravelTimeModel, radius_m: float = 400.0, refresh_min: float = 15.0,
                 max_steps: int | None = None) -> TripResult:
    """Replan bằng D* Lite: quan sát như `replan`, nhưng chỉ cập nhật cạnh thay đổi; dựng lại mỗi `refresh_min`."""
    t_start = time.perf_counter()
    delays: dict[int, float] = {}
    obs = ObservedModel(forecast, delays)
    h_scale = 0.995 / forecast.v_max_mpm
    u, t = source, t0
    edges: list[int] = []
    nodes = [source]
    expanded = rebuilds = updates = 0
    planner: DStarLite | None = None
    t_snap = t0
    max_steps = max_steps or 4 * graph.n_nodes
    lengths = graph.length
    planned_first = float("nan")
    while u != target and len(edges) < max_steps:
        new_obs = {}
        for e in graph.edges_within(u, radius_m).tolist():
            d = real.travel_time(e, t) - forecast.travel_time(e, t)
            if abs(d - delays.get(e, 0.0)) > 1e-9:
                new_obs[e] = d
        delays.update(new_obs)
        if planner is None or t - t_snap >= refresh_min:
            if planner is not None:
                expanded += planner.expanded
            t_snap = t
            planner = DStarLite(graph, u, target, snapshot_costs(obs.travel_time, graph.n_edges, t), h_scale)
            rebuilds += 1
        else:
            planner.move_to(u)
            for e in new_obs:
                planner.change_cost(e, max(forecast.travel_time(e, t_snap) + delays[e],
                                           float(lengths[e]) / forecast.v_max_mpm))
                updates += 1
        planner.compute_shortest_path()
        if math.isnan(planned_first):
            planned_first = planner.path_cost()
        e = planner.next_edge()
        if e is None:
            break
        t += real.travel_time(e, t)
        edges.append(e)
        u = graph.edge_to_list[e]
        nodes.append(u)
    if planner is not None:
        expanded += planner.expanded
    reached = u == target
    res = TripResult("replan_dstar", reached, edges, nodes, (t - t0) if reached else float("nan"), planned_first,
                     graph.path_length(edges), expanded, time.perf_counter() - t_start, rebuilds)
    res.extra = {"incremental_updates": updates}
    return res
