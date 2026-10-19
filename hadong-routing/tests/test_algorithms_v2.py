"""Test các cải tiến thuật toán: heuristic LB/ALT (C2), D* Lite (C1), replan lười (C3), Numba (C4), rủi ro (D2),
và mô hình nhiễu tương quan."""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.dijkstra import dijkstra
from src.algorithms.dstar_lite import DStarLite, replan_dstar
from src.algorithms.heuristics import ALT, lb_heuristic, min_travel_time
from src.algorithms.numba_td import td_astar_numba
from src.algorithms.replan import oracle, replan
from src.algorithms.risk import ScenarioBank, path_times, risk_route
from src.algorithms.td_astar import td_astar, td_dijkstra
from src.speed_model import Incident, TravelTimeModel, get_noise_model, real_V


@pytest.fixture(scope="module")
def base(graph, vbase):
    return TravelTimeModel(graph.length, vbase)


def test_noise_model_marginal_and_correlation(cfg, graph):
    nm = get_noise_model(graph, cfg)
    Z = nm.standard_field(3)
    assert abs(Z.std() - 1.0) < 0.05 and abs(Z.mean()) < 0.1
    if nm.kind == "correlated":
        lag = np.corrcoef(Z[:, :-1].ravel(), Z[:, 1:].ravel())[0, 1]
        assert lag > 0.3
        e = np.arange(0, graph.n_edges, 5)
        nxt = np.array([graph.out_of(int(graph.edge_to[i]))[0] for i in e])
        assert np.corrcoef(Z[e, 8], Z[nxt, 8])[0, 1] > 0.3


def test_real_V_correlated_fifo(cfg, graph, vbase):
    from src.speed_model import fifo_max_slope

    V = real_V(graph, vbase, 0.3, 11, cfg)
    assert fifo_max_slope(V, graph.length).max() <= 0.9 + 1e-9
    assert np.array_equal(V, real_V(graph, vbase, 0.3, 11, cfg))


def test_lb_and_alt_admissible_and_optimal(graph, vbase, base, od_pairs):
    tau_min = min_travel_time(graph.length, vbase)
    alt = ALT(graph, tau_min, n_landmarks=4)
    for i, (s, t) in enumerate(od_pairs[:10]):
        t0 = 480.0 + 37 * i
        ref = td_dijkstra(graph, s, t, t0, base)
        h_lb, _ = lb_heuristic(graph, t, tau_min)
        a_lb = td_astar(graph, s, t, t0, base, heuristic=h_lb)
        a_alt = td_astar(graph, s, t, t0, base, heuristic=alt.heuristic(t))
        a_eu = td_astar(graph, s, t, t0, base)
        assert a_lb.cost == pytest.approx(ref.cost, rel=1e-9)
        assert a_alt.cost == pytest.approx(ref.cost, rel=1e-9)
        assert a_lb.expanded <= a_eu.expanded
        assert h_lb[s] <= ref.cost + 1e-9 and alt.heuristic(t)[s] <= ref.cost + 1e-9


def test_dstar_lite_matches_dijkstra_after_changes(graph, base, od_pairs):
    s, t = od_pairs[2]
    cost = [base.travel_time(e, 480.0) for e in range(graph.n_edges)]
    ds = DStarLite(graph, s, t, list(cost), 0.995 / base.v_max_mpm)
    ds.compute_shortest_path()
    assert ds.path_cost() == pytest.approx(dijkstra(graph, s, t, cost).cost, rel=1e-9)
    # đi một bước, tăng mạnh chi phí vài cạnh của tuyến còn lại, kiểm tra lại
    e0 = ds.next_edge()
    s2 = graph.edge_to_list[e0]
    ds.move_to(s2)
    ref_path = dijkstra(graph, s2, t, cost).edges
    for e in ref_path[1:4]:
        cost[e] *= 20
        ds.change_cost(e, cost[e])
    ds.compute_shortest_path()
    assert ds.path_cost() == pytest.approx(dijkstra(graph, s2, t, cost).cost, rel=1e-9)


def test_replan_variants_reach_and_bounded(cfg, graph, vbase, base, od_pairs):
    for i, (s, t) in enumerate(od_pairs[:6]):
        t0 = 420.0 + 50 * i
        V = real_V(graph, vbase, 0.3, 200 + i, cfg)
        r0 = TravelTimeModel(graph.length, V)
        tmp = td_astar(graph, s, t, t0, base)
        real = TravelTimeModel(graph.length, V, [Incident(e, t0 - 5, 60, 0.15) for e in tmp.edges[2:4]])
        o = oracle(graph, s, t, t0, real)
        full = replan(graph, s, t, t0, base, real)
        lazy = replan(graph, s, t, t0, base, real, lazy_threshold=0.15)
        ds = replan_dstar(graph, s, t, t0, base, real, refresh_min=15)
        for r in (full, lazy, ds):
            assert r.reached and r.travel_time >= o.travel_time - 1e-9
        assert lazy.n_replans <= full.n_replans
        assert r0  # noqa: B015


def test_numba_matches_python(graph, base, od_pairs):
    for i, (s, t) in enumerate(od_pairs[:10]):
        t0 = 60.0 * (i * 2 % 24) + 13
        py = td_astar(graph, s, t, t0, base)
        nb = td_astar_numba(graph, s, t, t0, base)
        assert nb.cost == pytest.approx(py.cost, rel=1e-9)
        assert nb.expanded == py.expanded


def test_risk_route(cfg, graph, vbase, base, od_pairs):
    nm = get_noise_model(graph, cfg)
    bank = ScenarioBank(vbase, graph.length, nm, [0.3], 10, seed=5)
    s, t = od_pairs[4]
    po = td_astar(graph, s, t, 480.0, base)
    tt = path_times(bank.bank(0.3), graph.length, po.edges, 480.0)
    assert tt.shape == (10,) and np.all(tt > 0)
    # với kịch bản hệ số 1 (σ=0), path_times khớp mô phỏng scalar
    one = np.stack([vbase * 1000 / 60] * 2).astype(np.float64)
    assert path_times(one, graph.length, po.edges, 480.0)[0] == pytest.approx(po.cost, rel=1e-6)
    r = risk_route(graph, s, t, 480.0, base, base, bank.bank(0.3), k=3)
    assert r.reached and r.extra["planned_quantile"] > 0
