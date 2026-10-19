"""Giai đoạn 5 — môi trường Gymnasium cho định tuyến từng bước trên mạng giao thông động.

Agent đứng tại một nút, chọn một trong K cạnh ra (K = out-degree lớn nhất, có mask), di chuyển trong V_real.
Hai chế độ quan sát:
  * local — chỉ các cạnh ra của nút hiện tại (+ vị trí tương đối tới đích, giờ trong ngày) — đúng đặc tả ban đầu;
  * tree  — cây lân cận D bước (K + K² + … + K^D khe), mỗi khe là một cạnh với đặc trưng cục bộ; dùng cùng bộ
            trích đặc trưng GNN trên cây (src/rl/policies.py) — mục B1.
Định hình thưởng: "euclid" (giảm khoảng cách chim bay) hoặc "ctg" (thế năng = −cost-to-go thời gian tự do; chỉ
nằm trong THƯỞNG, chính sách không quan sát được) — mục B5.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.algorithms.dijkstra import dijkstra_all, freeflow_weights
from src.algorithms.td_astar import td_astar
from src.config import Config, deep_merge
from src.graph_data import RoadGraph
from src.scenarios import _corridor_edges, build_real_model
from src.speed_model import Incident, TravelTimeModel, base_model, real_V
from src.utils import derive_seed

N_GLOBAL = 6
TREE_FEAT = 8  # hợp lệ, v_obs, độ dài, cos tới đích, tiến độ (km), đã thăm, ngõ cụt, quay đầu


class RoutingEnv(gym.Env):
    """Môi trường định tuyến. `reset(options={"query": q})` chạy đúng truy vấn q (dùng khi đánh giá)."""

    metadata = {"render_modes": []}

    def __init__(self, graph: RoadGraph, V_base: np.ndarray, queries: list[dict[str, Any]], cfg: Config,
                 env_overrides: dict[str, Any] | None = None, level: int = 3, seed: int | None = None) -> None:
        super().__init__()
        self.graph, self.V_base, self.queries, self.cfg = graph, V_base, queries, cfg
        self.ec = deep_merge(cfg["env"], env_overrides or {})
        self.level = level
        self.K = graph.max_out_degree
        self.use_ctg = bool(self.ec["use_cost_to_go"])
        self.obs_mode = str(self.ec.get("obs_mode", "local"))
        self.depth = int(self.ec.get("tree_depth", 1)) if self.obs_mode == "tree" else 1
        self.shaping = str(self.ec["reward"].get("shaping", "euclid"))
        self.n_feat = (TREE_FEAT if self.obs_mode == "tree" else 5) + int(self.use_ctg)
        self.n_slots = sum(self.K ** d for d in range(1, self.depth + 1))
        self.n_global = N_GLOBAL + int(self.use_ctg)
        dim = self.n_global + self.n_slots * self.n_feat
        nb = [{v for _, v in graph.adj[u]} | {v for _, v in graph.radj[u]} for u in range(graph.n_nodes)]
        self.dead_end = np.array([len(x) <= 1 for x in nb])
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K)
        self.forecast = base_model(graph, V_base, cfg)
        self.ff_w = freeflow_weights(graph)
        self.v_norm_mpm = float(self.ec["v_norm_kmh"]) * 1000.0 / 60.0
        self._rng = np.random.default_rng(seed)
        self._hw = np.asarray(cfg["scenarios"]["hour_weights"], float)
        self._hw /= self._hw.sum()
        self._near = [q for q in queries if q["od_dist_m"] <= cfg["rl"]["curriculum"]["level1_max_dist_m"]] or queries
        self._ctg_cache: dict[int, np.ndarray] = {}
        self.real: TravelTimeModel = self.forecast
        self.u = self.dest = 0
        self.t = self.t0 = 0.0

    # ------------------------------------------------------------------
    def set_level(self, level: int) -> None:
        self.level = int(level)

    def _sample_episode(self) -> tuple[dict[str, Any], TravelTimeModel, float]:
        rng = self._rng
        cfg = self.cfg
        if self.level <= 1:
            q = self._near[int(rng.integers(len(self._near)))]
            return q, self.forecast, float(q["depart_min"])
        q = self.queries[int(rng.integers(len(self.queries)))]
        depart = float(rng.choice(24, p=self._hw) * 60 + rng.uniform(0, 60))
        sigma = float(rng.choice(cfg["scenarios"]["sigma_levels"]))
        V = real_V(self.graph, self.V_base, sigma, int(rng.integers(2**31)), cfg)
        incs: list[Incident] = []
        if self.level >= 3:
            sc = cfg["scenarios"]
            n_inc = int(rng.choice(sc["n_incidents_choices"], p=sc["n_incidents_probs"]))
            if n_inc:
                route = td_astar(self.graph, q["origin"], q["dest"], depart, self.forecast).edges
                corridor = _corridor_edges(self.graph, q["origin"], q["dest"], sc["incident_corridor_m"]).tolist()
                ic = cfg["speed"]["incidents"]
                lo, hi = sc["incident_start_offset_min"]
                for _ in range(n_inc):
                    pool = route if rng.random() < self.ec["train_incident_on_route_prob"] else corridor
                    if pool:
                        incs.append(Incident(int(rng.choice(pool)), depart + float(rng.uniform(lo, hi)),
                                             float(rng.uniform(*ic["duration_range_min"])),
                                             float(rng.uniform(*ic["severity_range"]))))
        return q, base_model(self.graph, V, cfg, incs), depart

    def _ctg(self, dest: int) -> np.ndarray:
        """Thời gian tự do còn lại (phút) từ mọi nút tới đích — chỉ dùng ở biến thể ablation rl_ctg."""
        if dest not in self._ctg_cache:
            if len(self._ctg_cache) > 2000:
                self._ctg_cache.clear()
            self._ctg_cache[dest] = dijkstra_all(self.graph, dest, self.ff_w, reverse=True)
        return self._ctg_cache[dest]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "query" in options:
            q = options["query"]
            self.real = build_real_model(self.graph, self.V_base, q, self.cfg,
                                         incident_key=options.get("incident_key", "incidents"))
            self.t0 = float(q["depart_min"])
        else:
            q, self.real, self.t0 = self._sample_episode()
        self.query = q
        self.u, self.dest, self.t = int(q["origin"]), int(q["dest"]), self.t0
        self.visited = {self.u}
        self.path_edges: list[int] = []
        self.steps = 0
        self.max_steps = int(self.ec["max_steps_factor"] * max(q["ref_hops"], 1) + self.ec["max_steps_slack"])
        self.ctg = self._ctg(self.dest) if (self.use_ctg or self.shaping == "ctg") else None
        self._dest_x, self._dest_y = self.graph.node_x[self.dest], self.graph.node_y[self.dest]
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _out(self) -> np.ndarray:
        return self.graph.out_of(self.u)

    def action_masks(self) -> np.ndarray:
        out = self._out()
        m = np.zeros(self.K, dtype=bool)
        m[: len(out)] = True
        if self.ec["mask_visited"]:
            vis = np.array([self.graph.edge_to_list[e] in self.visited and self.graph.edge_to_list[e] != self.dest
                            for e in out])
            if (~vis).any():
                m[: len(out)] = ~vis
        return m

    def _dist(self, n: int) -> float:
        return math.hypot(self.graph.node_x[n] - self._dest_x, self.graph.node_y[n] - self._dest_y)

    def _obs(self) -> np.ndarray:
        if self.obs_mode == "tree":
            return self._obs_tree()
        return self._obs_local()

    def _global(self) -> list[float]:
        g, ec = self.graph, self.ec
        ps = float(ec["pos_scale_m"])
        dx = (self._dest_x - g.node_x[self.u]) / ps
        dy = (self._dest_y - g.node_y[self.u]) / ps
        ang = 2 * math.pi * (self.t % 1440.0) / 1440.0
        glob = [dx, dy, math.hypot(dx, dy), math.sin(ang), math.cos(ang), self.steps / self.max_steps]
        if self.use_ctg:
            glob.append(self.ctg[self.u] / ec["ctg_scale_min"])
        return glob

    def tree_slots(self) -> list[np.ndarray]:
        """Chỉ số cạnh theo từng tầng của cây lân cận (−1 = khe trống). Tầng d có K^d khe (thứ tự BFS)."""
        g, K = self.graph, self.K
        levels = []
        heads = np.array([self.u])
        for _ in range(self.depth):
            slots = np.full(len(heads) * K, -1, dtype=np.int64)
            for i, h in enumerate(heads):
                if h < 0:
                    continue
                out = g.out_of(int(h))
                slots[i * K: i * K + len(out)] = out
            levels.append(slots)
            heads = np.where(slots >= 0, g.edge_to[np.maximum(slots, 0)], -1)
        return levels

    def _obs_tree(self) -> np.ndarray:
        g, ec = self.graph, self.ec
        levels = self.tree_slots()
        allslots = np.concatenate(levels)
        feats = np.zeros((len(allslots), self.n_feat), dtype=np.float64)
        valid = allslots >= 0
        es = allslots[valid]
        if len(es):
            tt = self.real.travel_time_vec(es, self.t)
            v_obs = g.length[es] / tt * np.clip(1.0 + ec["obs_speed_noise"] * self._rng.standard_normal(len(es)), 0.5, 1.5)
            fr, to = g.edge_from[es], g.edge_to[es]
            ex, ey = g.node_x[to] - g.node_x[fr], g.node_y[to] - g.node_y[fr]
            tx, ty = self._dest_x - g.node_x[fr], self._dest_y - g.node_y[fr]
            norm = np.hypot(ex, ey) * np.maximum(np.hypot(tx, ty), 1e-9)
            d_u = self._dist(self.u)
            d_to = np.hypot(g.node_x[to] - self._dest_x, g.node_y[to] - self._dest_y)
            # quay đầu: cạnh đi về đúng nút đầu của cạnh cha (chỉ xét từ tầng 2)
            parent_tail = np.full(len(allslots), -1, dtype=np.int64)
            off = 0
            for d, lv in enumerate(levels):
                if d > 0:
                    par = np.repeat(levels[d - 1], self.K)
                    parent_tail[off: off + len(lv)] = np.where(par >= 0, g.edge_from[np.maximum(par, 0)], -1)
                off += len(lv)
            f = feats[valid]
            f[:, 0] = 1.0
            f[:, 1] = v_obs / self.v_norm_mpm
            f[:, 2] = np.minimum(g.length[es] / ec["len_scale_m"], 10.0)
            f[:, 3] = np.where(norm > 0, (ex * tx + ey * ty) / np.maximum(norm, 1e-9), 0.0)
            f[:, 4] = (d_u - d_to) / 1000.0
            f[:, 5] = [1.0 if int(v) in self.visited else 0.0 for v in to]
            f[:, 6] = self.dead_end[to] & (to != self.dest)
            f[:, 7] = to == parent_tail[valid]
            if self.use_ctg:
                ff = np.asarray([self.ff_w[e] for e in es])
                f[:, 8] = np.maximum((self.ctg[fr] - ff - self.ctg[to]) / ec["ctg_delta_scale_min"], -10.0)
            feats[valid] = f
        obs = np.concatenate([np.asarray(self._global()), feats.ravel()]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _obs_local(self) -> np.ndarray:
        g, ec = self.graph, self.ec
        ps = float(ec["pos_scale_m"])
        dx = (self._dest_x - g.node_x[self.u]) / ps
        dy = (self._dest_y - g.node_y[self.u]) / ps
        dist = math.hypot(dx, dy)
        ang = 2 * math.pi * (self.t % 1440.0) / 1440.0
        glob = [dx, dy, dist, math.sin(ang), math.cos(ang), self.steps / self.max_steps]
        if self.use_ctg:
            glob.append(self.ctg[self.u] / ec["ctg_scale_min"])
        feats = np.zeros((self.K, self.n_feat), dtype=np.float64)
        out = self._out()
        if len(out):
            tt = self.real.travel_time_vec(out, self.t)
            v_obs = g.length[out] / tt
            v_obs = v_obs * np.clip(1.0 + ec["obs_speed_noise"] * self._rng.standard_normal(len(out)), 0.5, 1.5)
            to = g.edge_to[out]
            ex, ey = g.node_x[to] - g.node_x[self.u], g.node_y[to] - g.node_y[self.u]
            tx, ty = self._dest_x - g.node_x[self.u], self._dest_y - g.node_y[self.u]
            norm = np.hypot(ex, ey) * max(math.hypot(tx, ty), 1e-9)
            cosang = np.where(norm > 0, (ex * tx + ey * ty) / np.maximum(norm, 1e-9), 0.0)
            k = len(out)
            feats[:k, 0] = 1.0
            feats[:k, 1] = v_obs / self.v_norm_mpm
            feats[:k, 2] = np.minimum(g.length[out] / ec["len_scale_m"], 10.0)
            feats[:k, 3] = cosang
            feats[:k, 4] = [1.0 if int(v) in self.visited else 0.0 for v in to]
            if self.use_ctg:
                # độ "lệch" so với tuyến tự do tối ưu: 0 nếu cạnh nằm trên một tuyến tối ưu, âm nếu đi vòng
                ff = np.asarray([self.ff_w[e] for e in out])
                feats[:k, 5] = np.maximum((self.ctg[self.u] - ff - self.ctg[to]) / ec["ctg_delta_scale_min"], -10.0)
        obs = np.concatenate([np.asarray(glob), feats.ravel()]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def step(self, action: int):
        ec, rc = self.ec, self.ec["reward"]
        out = self._out()
        mask = self.action_masks()
        self.steps += 1
        a = int(action)
        if self.ec.get("invalid_action") == "wrap" and len(out):
            a = a % len(out)  # RecurrentPPO không hỗ trợ mask: ánh xạ mọi hành động về một cạnh hợp lệ
        if a >= len(out) or not mask[a]:  # hành động không hợp lệ (chỉ xảy ra khi không dùng mask, vd. DQN)
            pen = float(self.cfg["rl"]["dqn"]["invalid_action_penalty"])
            truncated = self.steps >= self.max_steps
            r = -pen - (rc["timeout_penalty"] if truncated else 0.0)
            return self._obs(), r, False, truncated, self._info(False)
        e = int(out[a])
        tt = float(self.real.travel_time_vec(np.array([e]), self.t)[0])
        d_before = self._dist(self.u)
        prev_u = self.u
        self.t += tt
        self.u = self.graph.edge_to_list[e]
        self.path_edges.append(e)
        d_after = self._dist(self.u)
        if self.shaping == "ctg":
            # thế năng Φ(s) = −ctg_tự_do(s) (phút): Φ(s') − Φ(s) = ctg(u) − ctg(v)
            shaping_min = float(self.ctg[prev_u] - self.ctg[self.u])
        else:
            shaping_min = (d_before - d_after) / 1000.0 / rc["shaping_ref_speed_kmh"] * 60.0
        r = (-rc["time_weight"] * tt + rc["shaping_lambda"] * shaping_min) / ec["time_scale_min"]
        if self.u in self.visited:
            r -= rc["revisit_penalty"]
        self.visited.add(self.u)
        terminated = truncated = False
        reached = self.u == self.dest
        if reached:
            r += rc["goal_bonus"]
            terminated = True
        elif not self.action_masks().any():
            r -= rc["deadend_penalty"]
            terminated = True
        elif self.steps >= self.max_steps:
            r -= rc["timeout_penalty"]
            truncated = True
        return self._obs(), float(r), terminated, truncated, self._info(reached)

    def _info(self, reached: bool) -> dict[str, Any]:
        return {"reached": reached, "travel_time": self.t - self.t0, "steps": self.steps,
                "query_id": self.query.get("id")}


def make_env_factory(graph: RoadGraph, V_base: np.ndarray, queries: list[dict[str, Any]], cfg: Config,
                     env_overrides: dict[str, Any] | None, seed: int, rank: int):
    """Factory cho VecEnv (mỗi env một seed con)."""

    def _f() -> RoutingEnv:
        return RoutingEnv(graph, V_base, queries, cfg, env_overrides, level=1, seed=derive_seed(seed, "env", rank))

    return _f
