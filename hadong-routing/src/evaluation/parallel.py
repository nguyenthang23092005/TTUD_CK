"""Bể tiến trình dùng chung: mỗi worker nạp đồ thị/V_base một lần (Windows dùng spawn)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable

_STATE: dict[str, Any] = {}


def _init(mode: str) -> None:
    from src.config import load_config
    from src.graph_data import RoadGraph
    from src.speed_model import base_model, load_vbase

    cfg = load_config(mode)
    graph = RoadGraph.from_config(cfg)
    V = load_vbase(cfg)
    _STATE.update(cfg=cfg, graph=graph, V=V, forecast=base_model(graph, V, cfg))


def state() -> dict[str, Any]:
    return _STATE


def pmap(fn: Callable[[Any], Any], items: Iterable[Any], mode: str, n_workers: int) -> list[Any]:
    """map song song có thứ tự; n_workers<=1 chạy tuần tự trong tiến trình hiện tại."""
    items = list(items)
    if n_workers <= 1:
        if not _STATE:
            _init(mode)
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init, initargs=(mode,)) as ex:
        return list(ex.map(fn, items, chunksize=max(1, len(items) // (4 * n_workers))))
