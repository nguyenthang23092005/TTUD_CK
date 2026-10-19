"""Test giai đoạn 5: môi trường Gymnasium (check_env, mask, kết thúc tập)."""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.td_astar import td_astar
from src.env.routing_env import RoutingEnv
from src.scenarios import build_real_model, generate


@pytest.fixture(scope="module")
def queries(cfg, graph, vbase):
    return generate(cfg, graph, vbase)


@pytest.mark.parametrize("ctg", [False, True])
def test_check_env(cfg, graph, vbase, queries, ctg):
    from gymnasium.utils.env_checker import check_env

    env = RoutingEnv(graph, vbase, queries["train"], cfg, {"use_cost_to_go": ctg}, level=3, seed=0)
    check_env(env, skip_render_check=True)


def test_mask_matches_out_degree(cfg, graph, vbase, queries):
    env = RoutingEnv(graph, vbase, queries["train"], cfg, level=2, seed=1)
    env.reset(seed=1)
    for _ in range(30):
        m = env.action_masks()
        deg = len(graph.out_of(env.u))
        assert m.sum() == deg and m[:deg].all() and not m[deg:].any()
        obs, r, term, trunc, info = env.step(int(np.flatnonzero(m)[0]))
        assert env.observation_space.contains(obs)
        # cờ hợp lệ trong quan sát khớp mask
        feats = obs[6:].reshape(env.K, env.n_feat)
        assert np.array_equal(feats[:, 0] > 0.5, env.action_masks())
        if term or trunc:
            env.reset()


def test_invalid_action_penalized(cfg, graph, vbase, queries):
    env = RoutingEnv(graph, vbase, queries["train"], cfg, level=1, seed=2)
    env.reset(seed=2)
    u = env.u
    deg = len(graph.out_of(u))
    if deg < env.K:
        _, r, _, _, _ = env.step(env.K - 1)
        assert r < 0 and env.u == u


def test_following_oracle_reaches_goal(cfg, graph, vbase, queries):
    q = queries["test"][0]
    env = RoutingEnv(graph, vbase, queries["train"], cfg, seed=3)
    env.reset(options={"query": q})
    real = build_real_model(graph, vbase, q, cfg)
    plan = td_astar(graph, q["origin"], q["dest"], q["depart_min"], real)
    info = {}
    for e in plan.edges:
        a = int(np.flatnonzero(graph.out_of(env.u) == e)[0])
        _, _, term, trunc, info = env.step(a)
    assert term and info["reached"]
    assert info["travel_time"] == pytest.approx(plan.cost, rel=1e-9)
