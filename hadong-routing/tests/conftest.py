"""Fixture dùng chung: nạp đồ thị/vận tốc đã sinh (bỏ qua test nếu chưa chạy giai đoạn 1–2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config("smoke")


@pytest.fixture(scope="session")
def graph(cfg):
    from src.graph_data import RoadGraph

    if not (cfg.path("processed_dir") / "graph.npz").exists():
        pytest.skip("Chưa có đồ thị — chạy `python -m src.graph_builder` trước")
    return RoadGraph.from_config(cfg)


@pytest.fixture(scope="session")
def vbase(cfg, graph):
    p = cfg.path("speeds_dir") / "V_base.npy"
    if not p.exists():
        pytest.skip("Chưa có V_base — chạy `python -m src.speed_model` trước")
    return np.load(p)


@pytest.fixture(scope="session")
def od_pairs(graph):
    rng = np.random.default_rng(123)
    pairs = []
    while len(pairs) < 25:
        s, t = rng.integers(0, graph.n_nodes, 2)
        if graph.straight_dist(int(s), int(t)) > 800:
            pairs.append((int(s), int(t)))
    return pairs
