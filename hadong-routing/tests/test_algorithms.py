"""Test giai đoạn 4: tính đúng của Dijkstra/A*/TD-A*/Yen/replan."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.algorithms.astar import astar_freeflow, astar_length
from src.algorithms.dijkstra import dijkstra, dijkstra_all, freeflow_weights, shortest_by_length
from src.algorithms.replan import oracle, plan_once, replan
from src.algorithms.td_astar import td_astar, td_dijkstra
from src.algorithms.yen import yen_k_shortest
from src.speed_model import Incident, TravelTimeModel, make_real_V


@pytest.fixture(scope="module")
def nxgraph(graph):
    G = nx.DiGraph()
    for e in range(graph.n_edges):
        G.add_edge(int(graph.edge_from[e]), int(graph.edge_to[e]), length=float(graph.length[e]))
    return G


def test_dijkstra_matches_networkx(graph, nxgraph, od_pairs):
    for s, t in od_pairs:
        r = shortest_by_length(graph, s, t)
        ref = nx.shortest_path_length(nxgraph, s, t, weight="length")
        assert r.found and r.cost == pytest.approx(ref, rel=1e-9)
        assert graph.path_length(r.edges) == pytest.approx(r.cost)
        assert r.nodes[0] == s and r.nodes[-1] == t


def test_dijkstra_all_matches(graph, nxgraph):
    d = dijkstra_all(graph, 10, graph.length_list)
    ref = nx.single_source_dijkstra_path_length(nxgraph, 10, weight="length")
    for v in list(ref)[:200]:
        assert d[v] == pytest.approx(ref[v])
    drev = dijkstra_all(graph, 10, graph.length_list, reverse=True)
    for v in list(ref)[:50]:
        assert drev[v] == pytest.approx(nx.shortest_path_length(nxgraph, v, 10, weight="length"))


def test_astar_optimal_and_fewer_expansions(graph, od_pairs):
    for s, t in od_pairs:
        d = shortest_by_length(graph, s, t)
        a = astar_length(graph, s, t)
        assert a.cost == pytest.approx(d.cost, rel=1e-9)
        assert a.expanded <= d.expanded
        ff_d = dijkstra(graph, s, t, freeflow_weights(graph))
        ff_a = astar_freeflow(graph, s, t)
        assert ff_a.cost == pytest.approx(ff_d.cost, rel=1e-9)
        assert ff_a.expanded <= ff_d.expanded


def test_td_astar_equals_td_dijkstra(graph, vbase, od_pairs):
    model = TravelTimeModel(graph.length, vbase)
    for i, (s, t) in enumerate(od_pairs):
        t0 = float(60 * (i % 24) + 17)
        d = td_dijkstra(graph, s, t, t0, model)
        a = td_astar(graph, s, t, t0, model)
        assert d.found and a.found
        assert a.cost == pytest.approx(d.cost, rel=1e-9, abs=1e-9)
        assert a.expanded <= d.expanded
        # thời gian tính lại bằng mô phỏng đúng bằng nhãn
        assert model.path_travel_time(a.edges, t0) == pytest.approx(a.cost, rel=1e-9)


def test_td_with_incidents_equal(graph, vbase, od_pairs):
    s, t = od_pairs[0]
    base = TravelTimeModel(graph.length, vbase)
    plan = td_astar(graph, s, t, 480.0, base)
    incs = [Incident(e, 470.0, 90.0, 0.1) for e in plan.edges[: max(1, len(plan.edges) // 3)]]
    real = TravelTimeModel(graph.length, make_real_V(vbase, graph.length, 0.2, 3), incs)
    d = td_dijkstra(graph, s, t, 480.0, real)
    a = td_astar(graph, s, t, 480.0, real)
    assert a.cost == pytest.approx(d.cost, rel=1e-9)


def test_fastest_not_slower_than_shortest(graph, vbase, od_pairs):
    model = TravelTimeModel(graph.length, vbase)
    for i, (s, t) in enumerate(od_pairs):
        t0 = float(60 * (i % 24))
        fast = td_astar(graph, s, t, t0, model)
        short = shortest_by_length(graph, s, t)
        assert fast.cost <= model.path_travel_time(short.edges, t0) + 1e-9
        assert graph.path_length(fast.edges) >= short.cost - 1e-6


def test_oracle_lower_bound(graph, vbase, od_pairs):
    base = TravelTimeModel(graph.length, vbase)
    for i, (s, t) in enumerate(od_pairs[:10]):
        t0 = 420.0 + 30 * i
        real = TravelTimeModel(graph.length, make_real_V(vbase, graph.length, 0.3, 100 + i))
        o = oracle(graph, s, t, t0, real)
        p = plan_once(graph, s, t, t0, base, real)
        r = replan(graph, s, t, t0, base, real, radius_m=400)
        assert o.reached and p.reached and r.reached
        assert o.travel_time <= p.travel_time + 1e-9
        assert o.travel_time <= r.travel_time + 1e-9
        assert r.nodes[-1] == t


def test_replan_without_noise_equals_plan_once(graph, vbase, od_pairs):
    base = TravelTimeModel(graph.length, vbase)
    s, t = od_pairs[3]
    p = plan_once(graph, s, t, 480.0, base, base)
    r = replan(graph, s, t, 480.0, base, base)
    assert r.travel_time == pytest.approx(p.travel_time, rel=1e-9)


def test_yen(graph, od_pairs):
    s, t = od_pairs[1]
    paths = yen_k_shortest(graph, s, t, graph.length_list, k=3)
    assert len(paths) == 3
    costs = [p.cost for p in paths]
    assert costs == sorted(costs)
    assert costs[0] == pytest.approx(shortest_by_length(graph, s, t).cost)
    assert len({tuple(p.edges) for p in paths}) == 3
    for p in paths:  # không lặp nút
        assert len(set(p.nodes)) == len(p.nodes)
        assert np.all(graph.edge_from[p.edges[1:]] == graph.edge_to[p.edges[:-1]])
