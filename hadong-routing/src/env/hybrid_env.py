"""Môi trường lai RL + tìm kiếm (mục B3).

Thay vì chọn từng cạnh, tác tử chọn hệ số β ∈ {0, 0.5, 1, 2, 4} — mức "tin" vào độ lệch quan sát δ_e (vận tốc thực
so với dự báo) trong bán kính quan sát. TD-A* lập kế hoạch trên τ_β(e,t) = max(τ_base(e,t) + β·δ_e, L_e/v_max) và xe đi
cạnh đầu tiên. β = 1 chính là replan; β = 0 là chỉ dùng dự báo (≈ plan-once); β > 1 là thận trọng (né mạnh vùng chậm).
Vì TD-A* luôn tìm được đường trên đồ thị liên thông mạnh nên tác tử LUÔN tới đích; RL chỉ học điều chỉnh mức tin cậy
theo ngữ cảnh (giờ, mức lệch quan sát, khoảng cách còn lại).
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.algorithms.replan import ObservedModel
from src.algorithms.td_astar import td_astar, td_heuristic
from src.config import Config, deep_merge
from src.env.routing_env import RoutingEnv
from src.graph_data import RoadGraph

N_HYB = 12


class ScaledDelays(dict):
    """dict δ nhân hệ số β khi đọc (không sao chép)."""

    def __init__(self, base: dict[int, float], beta: float) -> None:
        super().__init__()
        self.base, self.beta = base, beta

    def get(self, k: int, default: Any = None) -> Any:  # type: ignore[override]
        v = self.base.get(k)
        return default if v is None else self.beta * v


class HybridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, graph: RoadGraph, V_base: np.ndarray, queries: list[dict[str, Any]], cfg: Config,
                 env_overrides: dict[str, Any] | None = None, level: int = 3, seed: int | None = None) -> None:
        super().__init__()
        self.inner = RoutingEnv(graph, V_base, queries, cfg, env_overrides, level=level, seed=seed)
        self.graph, self.cfg = graph, cfg
        self.ec = deep_merge(cfg["env"], env_overrides or {})
        self.betas = [float(b) for b in self.ec["hybrid"]["betas"]]
        self.radius = float(cfg["algorithms"]["replan"]["observation_radius_m"])
        self.action_space = spaces.Discrete(len(self.betas))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(N_HYB,), dtype=np.float32)

    def set_level(self, level: int) -> None:
        self.inner.set_level(level)

    def action_masks(self) -> np.ndarray:
        return np.ones(len(self.betas), dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.inner.reset(seed=seed, options=options)
        e = self.inner
        self.forecast = e.forecast
        self.delays: dict[int, float] = {}
        self.h = td_heuristic(self.graph, e.dest, self.forecast.v_max_mpm)
        self.ref_time = max(float(e.query.get("ref_time_base_min", 10.0)), 1.0)
        self.last_beta = 1.0
        self.max_steps = e.max_steps
        self._observe()
        return self._obs(), {}

    def _observe(self) -> None:
        e = self.inner
        self.new_rel: list[float] = []
        for ed in self.graph.edges_within(e.u, self.radius).tolist():
            tb = self.forecast.travel_time(ed, e.t)
            d = e.real.travel_time(ed, e.t) - tb
            self.delays[ed] = d
            self.new_rel.append(d / max(tb, 1e-6))

    def _obs(self) -> np.ndarray:
        e = self.inner
        g = self.graph
        ps = float(self.ec["pos_scale_m"])
        rel = np.asarray(self.new_rel) if self.new_rel else np.zeros(1)
        all_rel = np.asarray(list(self.delays.values())) if self.delays else np.zeros(1)
        ang = 2 * math.pi * (e.t % 1440.0) / 1440.0
        dist = math.hypot(g.node_x[e.u] - g.node_x[e.dest], g.node_y[e.u] - g.node_y[e.dest]) / ps
        obs = [dist, math.sin(ang), math.cos(ang), e.steps / max(self.max_steps, 1),
               float(np.mean(rel)), float(np.max(rel)), float(np.mean(rel > 0.5)), float(np.mean(rel < -0.2)),
               float(np.clip(np.mean(all_rel), -10, 10)), (e.t - e.t0) / self.ref_time, self.last_beta / 4.0,
               len(self.new_rel) / 50.0]
        return np.clip(np.asarray(obs, dtype=np.float32), -10.0, 10.0)

    def step(self, action: int):
        e = self.inner
        beta = self.betas[int(action)]
        self.last_beta = beta
        model = ObservedModel(self.forecast, ScaledDelays(self.delays, beta), self.forecast.v_max_mpm)
        plan = td_astar(self.graph, e.u, e.dest, e.t, model, heuristic=self.h)
        ed = plan.edges[0]
        tt = float(e.real.travel_time(ed, e.t))
        e.t += tt
        e.u = self.graph.edge_to_list[ed]
        e.path_edges.append(ed)
        e.steps += 1
        reached = e.u == e.dest
        r = -tt / self.ec["time_scale_min"]
        truncated = (not reached) and e.steps >= self.max_steps
        if not reached:
            self._observe()
        info = {"reached": reached, "travel_time": e.t - e.t0, "steps": e.steps, "query_id": e.query.get("id"),
                "beta": beta}
        return self._obs(), float(r), reached, truncated, info

    @property
    def path_edges(self) -> list[int]:
        return self.inner.path_edges


def make_hybrid_factory(graph: RoadGraph, V_base: np.ndarray, queries: list[dict[str, Any]], cfg: Config,
                        env_overrides: dict[str, Any] | None, seed: int, rank: int):
    from src.utils import derive_seed

    def _f() -> HybridEnv:
        return HybridEnv(graph, V_base, queries, cfg, env_overrides, level=1, seed=derive_seed(seed, "henv", rank))

    return _f
