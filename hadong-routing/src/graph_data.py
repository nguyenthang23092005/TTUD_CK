"""Lớp RoadGraph: nạp đồ thị đã xử lý (CSR numpy + bảng cạnh) cho thuật toán, môi trường RL và ứng dụng."""
from __future__ import annotations

import math
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import Config
from src.utils import local_xy_m


class RoadGraph:
    """Đồ thị có hướng dạng CSR. Chỉ số nút 0..N-1, chỉ số cạnh 0..E-1."""

    def __init__(self, processed_dir: Path) -> None:
        z = np.load(processed_dir / "graph.npz", allow_pickle=False)
        self.out_ptr: np.ndarray = z["out_ptr"]
        self.out_edges: np.ndarray = z["out_edges"]
        self.edge_from: np.ndarray = z["edge_from"]
        self.edge_to: np.ndarray = z["edge_to"]
        self.length: np.ndarray = z["length"]
        self.v_free: np.ndarray = z["v_free"]
        self.group_id: np.ndarray = z["group_id"]
        self.group_names: list[str] = [str(x) for x in z["group_names"]]
        self.node_lat: np.ndarray = z["node_lat"]
        self.node_lon: np.ndarray = z["node_lon"]
        self.node_osmid: np.ndarray = z["node_osmid"]
        self.bearing: np.ndarray = z["bearing"]
        self.n_nodes: int = len(self.node_lat)
        self.n_edges: int = len(self.edge_to)
        self._edges_path = processed_dir / "edges.parquet"
        self.lat0 = float(self.node_lat.mean())
        self.lon0 = float(self.node_lon.mean())
        self.node_x, self.node_y = local_xy_m(self.node_lat, self.node_lon, self.lat0, self.lon0)
        # danh sách Python cho vòng lặp nhanh trong heapq
        self.length_list: list[float] = self.length.tolist()
        self.edge_to_list: list[int] = self.edge_to.tolist()
        oe = self.out_edges.tolist()
        ptr = self.out_ptr.tolist()
        self.adj: list[list[tuple[int, int]]] = [
            [(e, self.edge_to_list[e]) for e in oe[ptr[u]:ptr[u + 1]]] for u in range(self.n_nodes)]
        self.radj: list[list[tuple[int, int]]] = [[] for _ in range(self.n_nodes)]
        for e, (u, v) in enumerate(zip(self.edge_from.tolist(), self.edge_to_list)):
            self.radj[v].append((e, u))
        self.max_out_degree: int = int(np.diff(self.out_ptr).max())
        self._kdtree = cKDTree(np.column_stack([self.node_x, self.node_y]))
        self._radius_cache: dict[tuple[int, float], np.ndarray] = {}
        self._edge_index: dict[tuple[int, int], int] = {
            (int(u), int(v)): i for i, (u, v) in enumerate(zip(self.edge_from, self.edge_to))}

    @classmethod
    def from_config(cls, cfg: Config) -> "RoadGraph":
        return cls(cfg.path("processed_dir"))

    @cached_property
    def edges_df(self) -> pd.DataFrame:
        return pd.read_parquet(self._edges_path)

    @cached_property
    def edge_names(self) -> list[str]:
        return self.edges_df["name"].fillna("").tolist()

    @cached_property
    def edge_coords(self) -> list[np.ndarray]:
        """Toạ độ (lat, lon) dọc hình học của từng cạnh."""
        from shapely import wkt

        return [np.asarray(wkt.loads(w).coords)[:, ::-1] for w in self.edges_df["geometry_wkt"]]

    def out_of(self, u: int) -> np.ndarray:
        return self.out_edges[self.out_ptr[u]:self.out_ptr[u + 1]]

    def edge_id(self, u: int, v: int) -> int:
        return self._edge_index[(u, v)]

    def nearest_node(self, lat: float, lon: float) -> int:
        x, y = local_xy_m(lat, lon, self.lat0, self.lon0)
        return int(self._kdtree.query([float(x), float(y)])[1])

    def nearest_edge(self, lat: float, lon: float) -> int:
        """Cạnh có trung điểm gần (lat, lon) nhất."""
        df = self.edges_df
        x, y = local_xy_m(lat, lon, self.lat0, self.lon0)
        mx, my = local_xy_m(df["mid_lat"].to_numpy(), df["mid_lon"].to_numpy(), self.lat0, self.lon0)
        return int(np.argmin((mx - x) ** 2 + (my - y) ** 2))

    def edges_within(self, node: int, radius_m: float) -> np.ndarray:
        """Các cạnh có nút đầu nằm trong bán kính `radius_m` quanh `node` (có cache)."""
        key = (node, radius_m)
        if key not in self._radius_cache:
            nbrs = self._kdtree.query_ball_point([self.node_x[node], self.node_y[node]], radius_m)
            self._radius_cache[key] = (np.concatenate([self.out_of(n) for n in nbrs])
                                       if nbrs else np.empty(0, dtype=np.int64))
        return self._radius_cache[key]

    def path_length(self, edges: list[int]) -> float:
        return float(sum(self.length_list[e] for e in edges))

    def straight_dist(self, a: int, b: int) -> float:
        return float(math.hypot(self.node_x[a] - self.node_x[b], self.node_y[a] - self.node_y[b]))

    def dist_to(self, target: int) -> np.ndarray:
        """Khoảng cách chim bay (m) từ mọi nút tới `target` (xấp xỉ phẳng, sai số < 0.1% ở quy mô quận)."""
        return np.hypot(self.node_x - self.node_x[target], self.node_y - self.node_y[target])
