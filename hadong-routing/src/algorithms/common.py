"""Kiểu dữ liệu và tiện ích dùng chung cho các thuật toán tìm đường."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


class TimeModel(Protocol):
    """Giao diện tối thiểu của một hàm thời gian qua cạnh phụ thuộc thời gian."""

    v_max_mpm: float

    def travel_time(self, e: int, t: float) -> float: ...


@dataclass
class SearchResult:
    """Kết quả một lần tìm đường.

    cost: tổng trọng số (m với trọng số chiều dài, phút với trọng số thời gian).
    arrival: thời điểm đến (phút) — chỉ với thuật toán phụ thuộc thời gian.
    """

    found: bool
    nodes: list[int] = field(default_factory=list)
    edges: list[int] = field(default_factory=list)
    cost: float = float("inf")
    arrival: float | None = None
    expanded: int = 0
    runtime_s: float = 0.0


def reconstruct(pred_edge: dict[int, int], edge_from: Sequence[int], source: int, target: int) -> tuple[list[int], list[int]]:
    """Dựng lại tuyến từ bảng cạnh tiền nhiệm (pred_edge[v] = cạnh đi vào v)."""
    edges: list[int] = []
    v = target
    while v != source:
        e = pred_edge[v]
        edges.append(e)
        v = int(edge_from[e])
    edges.reverse()
    nodes = [source] + [0] * len(edges)
    return edges, nodes


def nodes_from_edges(source: int, edges: Sequence[int], edge_to: Sequence[int]) -> list[int]:
    return [source] + [int(edge_to[e]) for e in edges]
