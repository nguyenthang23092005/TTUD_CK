"""Test giai đoạn 1 (đồ thị) và 2 (vận tốc, FIFO, sự cố)."""
from __future__ import annotations

import numpy as np
import pytest

from src.graph_builder import highway_group, parse_maxspeed
from src.speed_model import (Incident, TravelTimeModel, check_fifo_numeric, enforce_fifo, fifo_max_slope,
                             hourly_profile, make_real_V)


@pytest.mark.parametrize("raw,expected", [("50", 50.0), ("50 mph", 50 * 1.609344), (["60", "50"], 50.0),
                                          ("['60', '50']", 50.0), ("40;60", 40.0), (None, None), ("none", None)])
def test_parse_maxspeed(raw, expected):
    got = parse_maxspeed(raw)
    assert (got is None and expected is None) or got == pytest.approx(expected)


def test_highway_group(cfg):
    groups = cfg["graph"]["highway_groups"]
    assert highway_group("trunk_link", groups) == "trunk"
    assert highway_group(["residential", "tertiary"], groups) == "tertiary"
    assert highway_group("['secondary', 'residential']", groups) == "secondary"
    assert highway_group("weird_type", groups) == "residential"


def test_graph_csr_consistent(graph):
    assert graph.out_ptr[-1] == graph.n_edges
    for u in range(0, graph.n_nodes, 97):
        for e in graph.out_of(u):
            assert graph.edge_from[e] == u
    assert np.all(graph.length > 0)
    assert np.all(graph.v_free > 0)
    # chiều dài cạnh >= khoảng cách chim bay (điều kiện admissible của heuristic)
    straight = np.hypot(graph.node_x[graph.edge_from] - graph.node_x[graph.edge_to],
                        graph.node_y[graph.edge_from] - graph.node_y[graph.edge_to])
    assert np.all(graph.length >= straight * 0.995)
    assert not np.any(graph.edge_from == graph.edge_to)


def test_graph_strongly_connected(graph):
    import networkx as nx

    G = nx.DiGraph(zip(graph.edge_from.tolist(), graph.edge_to.tolist()))
    assert nx.is_strongly_connected(G)


def test_profile_periodic_and_peaks(cfg):
    p = hourly_profile(cfg)
    assert np.allclose(hourly_profile(cfg, np.array([0.0])), hourly_profile(cfg, np.array([24.0])))
    assert p[8] > p[3] and p[18] > p[3] and p[8] > p[13] and p[18] > p[13]


def test_vbase_bounds(cfg, graph, vbase):
    assert vbase.shape == (graph.n_edges, 24)
    assert vbase.min() >= cfg["speed"]["v_min_kmh"] - 1e-9
    assert np.all(vbase <= graph.v_free[:, None] + 1e-6)


def test_vbase_fifo_analytic_and_numeric(cfg, graph, vbase):
    assert fifo_max_slope(vbase, graph.length).max() <= cfg["speed"]["fifo"]["max_slope"] + 1e-9
    model = TravelTimeModel(graph.length, vbase)
    worst = np.argsort(-fifo_max_slope(vbase, graph.length).max(axis=1))[:50]
    assert check_fifo_numeric(model, worst, np.arange(0.0, 1440.0, 0.5)) <= 1e-9


def test_enforce_fifo_repairs_violation():
    L = np.array([3000.0])
    V = np.full((1, 24), 60.0)
    V[0, 5] = 5.0  # tăng vọt 5 -> 60 km/h trong 1 giờ trên cạnh 3 km: vi phạm FIFO
    assert fifo_max_slope(V, L).max() > 1.0
    V2, n = enforce_fifo(V, L, 0.9)
    assert n > 0 and fifo_max_slope(V2, L).max() <= 0.9 + 1e-9
    assert np.all(V2 >= V - 1e-12)  # chỉ tăng vận tốc


def test_incident_fifo_and_slowdown(cfg, graph, vbase):
    e = int(np.argmax(graph.length))
    inc = Incident(edge=e, start_min=480.0, duration_min=60.0, severity=0.1)
    m0 = TravelTimeModel(graph.length, vbase)
    m1 = TravelTimeModel(graph.length, vbase, [inc], ramp_in=5, ramp_out=15)
    assert m1.travel_time(e, 500.0) > 5 * m0.travel_time(e, 500.0)
    assert m1.travel_time(e, 300.0) == pytest.approx(m0.travel_time(e, 300.0))
    assert check_fifo_numeric(m1, [e], np.arange(400.0, 700.0, 0.25)) <= 1e-9


def test_real_V_reproducible_and_fifo(cfg, graph, vbase):
    a = make_real_V(vbase, graph.length, 0.3, seed=7)
    b = make_real_V(vbase, graph.length, 0.3, seed=7)
    assert np.array_equal(a, b)
    assert fifo_max_slope(a, graph.length).max() <= 0.9 + 1e-9


def test_vectorized_matches_scalar(graph, vbase):
    m = TravelTimeModel(graph.length, vbase)
    edges = np.arange(0, graph.n_edges, 37)
    for t in (0.0, 59.5, 487.3, 1439.9, 1500.0):
        vec = m.travel_time_vec(edges, t)
        sca = np.array([m.travel_time(int(e), t) for e in edges])
        assert np.allclose(vec, sca)
