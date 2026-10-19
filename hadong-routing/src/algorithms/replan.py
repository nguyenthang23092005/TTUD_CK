"""Mô phỏng chuyến đi trong môi trường thực V_real: plan-once, replan (TD-A* lặp tại mỗi giao lộ), oracle,
và tuyến tĩnh (Dijkstra khoảng cách / thời gian tự do) đem đi chạy trong V_real.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.algorithms.common import SearchResult
from src.algorithms.td_astar import td_astar, td_heuristic
from src.graph_data import RoadGraph
from src.speed_model import TravelTimeModel


@dataclass
class TripResult:
    """Kết quả một chuyến đi mô phỏng trong V_real."""

    method: str
    reached: bool
    edges: list[int] = field(default_factory=list)
    nodes: list[int] = field(default_factory=list)
    travel_time: float = float("nan")      # phút, trong V_real
    planned_time: float = float("nan")     # phút, theo mô hình dùng để lập kế hoạch
    length_m: float = float("nan")
    expanded: int = 0
    runtime_s: float = 0.0
    n_replans: int = 0
    extra: dict = field(default_factory=dict)


def simulate(graph: RoadGraph, real: TravelTimeModel, edges: Sequence[int], t0: float) -> float:
    """Đi theo dãy cạnh trong môi trường thực; trả về thời gian (phút)."""
    return real.path_travel_time(edges, t0)


def trip_from_plan(method: str, graph: RoadGraph, plan: SearchResult, real: TravelTimeModel, t0: float,
                   source: int) -> TripResult:
    if not plan.found:
        return TripResult(method, False, expanded=plan.expanded, runtime_s=plan.runtime_s)
    return TripResult(method, True, plan.edges, plan.nodes, simulate(graph, real, plan.edges, t0), plan.cost,
                      graph.path_length(plan.edges), plan.expanded, plan.runtime_s)


def plan_once(graph: RoadGraph, source: int, target: int, t0: float, forecast: TravelTimeModel,
              real: TravelTimeModel, safety: float = 0.995) -> TripResult:
    """TD-A* một lần trên dự báo V_base rồi đi theo kế hoạch trong V_real."""
    return trip_from_plan("plan_once", graph, td_astar(graph, source, target, t0, forecast, safety=safety), real, t0, source)


def oracle(graph: RoadGraph, source: int, target: int, t0: float, real: TravelTimeModel,
           safety: float = 0.995) -> TripResult:
    """TD-A* chạy thẳng trên V_real (biết trước tương lai) — cận dưới."""
    return trip_from_plan("oracle", graph, td_astar(graph, source, target, t0, real, safety=safety), real, t0, source)


class ObservedModel:
    """Mô hình dự báo + quan sát: τ_obs(e,t) = max(τ_base(e,t) + δ_e, L_e / v_max).

    δ_e = τ_real(e, t_quan_sát) − τ_base(e, t_quan_sát) là độ lệch đo được tại thời điểm quan sát và được giả
    định duy trì (cộng hằng số giữ nguyên tính FIFO). Cận dưới L/v_max giữ heuristic admissible.
    """

    def __init__(self, base: TravelTimeModel, delays: dict[int, float], v_max_mpm: float | None = None,
                 ref: TravelTimeModel | None = None) -> None:
        """base: mô hình cho cạnh CHƯA quan sát; ref: mô hình mà δ được đo so với (mặc định = base)."""
        self.base = base
        self.ref = ref if ref is not None else base
        self.delays = delays
        # v_max của heuristic: giữ bằng v_max của dự báo gốc để heuristic tính sẵn vẫn admissible
        self.v_max_mpm = v_max_mpm if v_max_mpm is not None else base.v_max_mpm
        self._len = base._len
        self._clamp_base = self.base is not self.ref

    def travel_time(self, e: int, t: float) -> float:
        d = self.delays.get(e)
        if d is None:
            tb = self.base.travel_time(e, t)
            return max(tb, self._len[e] / self.v_max_mpm) if self._clamp_base else tb
        return max(self.ref.travel_time(e, t) + d, self._len[e] / self.v_max_mpm)


def replan(graph: RoadGraph, source: int, target: int, t0: float, forecast: TravelTimeModel,
           real: TravelTimeModel, radius_m: float = 400.0, remember: bool = True, safety: float = 0.995,
           max_steps: int | None = None, lazy_threshold: float | None = None,
           base_provider: Callable[[int, float, dict[int, float]], TravelTimeModel] | None = None,
           method: str = "replan") -> TripResult:
    """Tại mỗi giao lộ: quan sát vận tốc thực các cạnh trong bán kính rồi chạy lại TD-A* từ vị trí hiện tại.

    lazy_threshold (mục C3): nếu không None, chỉ chạy lại khi một cạnh quan sát có |δ_mới − δ_lúc_lập_kế_hoạch|
        > lazy_threshold · τ_base(e); ngược lại đi tiếp theo kế hoạch cũ.
    base_provider (mục D1): hàm (nút, t, delays) -> mô hình dự báo mới (ví dụ GCN điền các cạnh chưa quan sát);
        mặc định dùng `forecast` cố định.
    """
    t_start = time.perf_counter()
    h = td_heuristic(graph, target, forecast.v_max_mpm, safety)
    delays: dict[int, float] = {}
    u, t = source, t0
    edges: list[int] = []
    nodes = [source]
    expanded = n_replans = 0
    planned_first = float("nan")
    max_steps = max_steps or 4 * graph.n_nodes
    plan_edges: list[int] = []
    planned_delays: dict[int, float] = {}
    while u != target and len(edges) < max_steps:
        if not remember:
            delays.clear()
        need = not plan_edges or lazy_threshold is None
        for e in graph.edges_within(u, radius_m).tolist():
            d = real.travel_time(e, t) - forecast.travel_time(e, t)
            delays[e] = d
            if not need and abs(d - planned_delays.get(e, 0.0)) > lazy_threshold * forecast.travel_time(e, t):
                need = True
        if need:
            base = base_provider(u, t, delays) if base_provider is not None else forecast
            plan = td_astar(graph, u, target, t, ObservedModel(base, delays, forecast.v_max_mpm, ref=forecast),
                            heuristic=h)
            n_replans += 1
            expanded += plan.expanded
            if not plan.found:  # không thể xảy ra trên đồ thị liên thông mạnh
                break
            if n_replans == 1:
                planned_first = plan.cost
            plan_edges = list(plan.edges)
            planned_delays = dict(delays)
        e = plan_edges.pop(0)
        t += real.travel_time(e, t)
        edges.append(e)
        u = graph.edge_to_list[e]
        nodes.append(u)
    reached = u == target
    return TripResult(method, reached, edges, nodes, (t - t0) if reached else float("nan"), planned_first,
                      graph.path_length(edges), expanded, time.perf_counter() - t_start, n_replans)


def static_trip(method: str, graph: RoadGraph, plan: SearchResult, real: TravelTimeModel, t0: float,
                source: int) -> TripResult:
    """Tuyến tĩnh (ví dụ Dijkstra khoảng cách) đem đi trong V_real."""
    return trip_from_plan(method, graph, plan, real, t0, source)
