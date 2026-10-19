"""Test các thành phần cải tiến: quan sát cây + GNN (B1), môi trường lai (B3), định hình ctg (B5),
dự báo GCN (D1), mô phỏng nhiều xe (E3)."""
from __future__ import annotations

import numpy as np
import pytest

from src.scenarios import generate


@pytest.fixture(scope="module")
def queries(cfg, graph, vbase):
    return generate(cfg, graph, vbase)


def test_tree_env_check_and_layout(cfg, graph, vbase, queries):
    from gymnasium.utils.env_checker import check_env

    from src.env.routing_env import RoutingEnv

    env = RoutingEnv(graph, vbase, queries["train"], cfg, {"obs_mode": "tree", "tree_depth": 2}, level=3, seed=0)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1)
    K = env.K
    assert obs.shape == (env.n_global + (K + K * K) * env.n_feat,)
    lv = env.tree_slots()
    assert len(lv) == 2 and len(lv[1]) == K * K
    # cờ hợp lệ tầng 1 khớp mask hành động
    f1 = obs[env.n_global: env.n_global + K * env.n_feat].reshape(K, env.n_feat)
    assert np.array_equal(f1[:, 0] > 0.5, env.action_masks())
    # con của khe j tầng 2 là cạnh ra của nút đầu cạnh j
    for j, e in enumerate(lv[0]):
        if e >= 0:
            kids = lv[1][j * K:(j + 1) * K]
            assert set(kids[kids >= 0].tolist()) == set(graph.out_of(int(graph.edge_to[e])).tolist())


def test_tree_gnn_extractor_forward(cfg, graph, vbase, queries):
    import torch

    from src.env.routing_env import RoutingEnv
    from src.rl.policies import TreeGNNExtractor

    env = RoutingEnv(graph, vbase, queries["train"], cfg, {"obs_mode": "tree", "tree_depth": 3}, seed=0)
    obs, _ = env.reset(seed=2)
    ext = TreeGNNExtractor(env.observation_space, env.n_global, env.K, env.depth, env.n_feat, hidden=16)
    out = ext(torch.as_tensor(np.stack([obs, obs])))
    assert out.shape == (2, 16 + env.K * 16) and torch.isfinite(out).all()


def test_ctg_shaping_is_potential_based(cfg, graph, vbase, queries):
    """Tổng shaping trên một quỹ đạo = ctg(gốc) − ctg(đích) (= ctg(gốc)), không phụ thuộc đường đi."""
    from src.env.routing_env import RoutingEnv

    env = RoutingEnv(graph, vbase, queries["train"], cfg, {"reward": {"shaping": "ctg", "goal_bonus": 0.0,
                     "revisit_penalty": 0.0, "time_weight": 0.0}}, seed=0)
    q = queries["test"][0]
    env.reset(options={"query": q})
    from src.algorithms.td_astar import td_astar

    plan = td_astar(graph, q["origin"], q["dest"], q["depart_min"], env.real)
    total = 0.0
    for e in plan.edges:
        _, r, *_ = env.step(int(np.flatnonzero(graph.out_of(env.u) == e)[0]))
        total += r
    assert total == pytest.approx(env.ctg[q["origin"]] * env.ec["reward"]["shaping_lambda"] / env.ec["time_scale_min"], rel=1e-6)


def test_hybrid_env_always_reaches(cfg, graph, vbase, queries):
    from gymnasium.utils.env_checker import check_env

    from src.env.hybrid_env import HybridEnv

    env = HybridEnv(graph, vbase, queries["train"], cfg, level=3, seed=0)
    check_env(env, skip_render_check=True)
    for i, q in enumerate(queries["test"][:3]):
        env.reset(options={"query": q})
        done, info = False, {}
        while not done:
            _, _, term, trunc, info = env.step(i % len(env.betas))
            done = term or trunc
        assert info["reached"]


def test_wrap_actions_for_recurrent(cfg, graph, vbase, queries):
    from src.env.routing_env import RoutingEnv

    env = RoutingEnv(graph, vbase, queries["train"], cfg, {"invalid_action": "wrap"}, seed=0)
    env.reset(seed=0)
    u = env.u
    env.step(env.K - 1)
    assert env.u != u  # luôn di chuyển, không bị phạt đứng yên


def test_forecaster_shapes(cfg, graph, vbase):
    from src.forecast import Forecaster, sensor_mask

    m = sensor_mask(graph, cfg)
    assert abs(m.mean() - cfg["forecast"]["sensor_fraction"]) < 0.02
    fc = Forecaster(graph, vbase, cfg)
    pred = fc.predict(np.zeros(graph.n_edges), m, 480.0)
    assert pred.shape == (graph.n_edges, 2) and np.isfinite(pred).all()
    V = fc.forecast_V(np.zeros((graph.n_edges, 2)), 480.0)
    assert np.allclose(V, vbase)  # log-tỉ số 0 ⇒ trùng dự báo nền


def test_multiagent_load_model(cfg, graph, vbase):
    from src.evaluation.multiagent import LoadModel, evaluate_paths
    from src.speed_model import TravelTimeModel

    base = TravelTimeModel(graph.length, vbase)
    mc = cfg["multiagent"]
    cap = np.full(graph.n_edges, 600.0)
    load = np.zeros((graph.n_edges, 192))
    load[5, 32] = 10
    lm = LoadModel(base, load, cap, mc)
    assert lm.travel_time(5, 32 * mc["slot_min"] + 1) > base.travel_time(5, 32 * mc["slot_min"] + 1)
    assert lm.travel_time(6, 480.0) == base.travel_time(6, 480.0)
    from src.algorithms.td_astar import td_astar

    p = td_astar(graph, 0, int(graph.edge_to[graph.out_of(0)[0]]), 480.0, base).edges
    tts, _ = evaluate_paths(base, [p] * 50, [480.0] * 50, cap, mc, 192)
    assert tts[0] >= base.path_travel_time(p, 480.0)
