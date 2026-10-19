"""Test giai đoạn 3: tách theo cặp O–D, ràng buộc khoảng cách, tái lập."""
from __future__ import annotations

import pytest

from src.scenarios import build_real_model, generate, load_split


@pytest.fixture(scope="module")
def sets(cfg, graph, vbase):
    return generate(cfg, graph, vbase)


def test_split_disjoint_by_od(sets):
    pairs = {s: {(q["origin"], q["dest"]) for q in qs} for s, qs in sets.items()}
    assert not (pairs["train"] & pairs["test"])
    assert not (pairs["train"] & pairs["val"])
    assert not (pairs["val"] & pairs["test"])


def test_constraints(cfg, sets):
    for qs in sets.values():
        for q in qs:
            assert q["od_dist_m"] >= cfg["scenarios"]["min_od_dist_m"]
            assert 0 <= q["depart_min"] < 1440
            assert q["ref_hops"] > 0


def test_test_incidents_unseen(sets):
    keys = ("incidents", "incidents_stress")
    train_edges = {i["edge"] for q in sets["train"] for k in keys for i in q[k]}
    test_edges = {i["edge"] for q in sets["test"] for k in keys for i in q[k]}
    assert not (train_edges & test_edges)


def test_main_and_stress_share_timing(sets):
    for q in sets["test"]:
        assert len(q["incidents"]) <= len(q["incidents_stress"]) + 3
        for a, b in zip(q["incidents"], q["incidents_stress"]):
            assert a["start_min"] == b["start_min"] and a["severity"] == b["severity"]


def test_reproducible(cfg, graph, vbase, sets):
    again = generate(cfg, graph, vbase)
    assert again == sets


def test_real_model_reproducible(cfg, graph, vbase, sets):
    q = sets["test"][0]
    a = build_real_model(graph, vbase, q, cfg)
    b = build_real_model(graph, vbase, q, cfg)
    assert a.travel_time(5, q["depart_min"]) == b.travel_time(5, q["depart_min"])
