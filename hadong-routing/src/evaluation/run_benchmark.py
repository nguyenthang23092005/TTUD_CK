"""Giai đoạn 7 — chạy mọi phương pháp cổ điển trên test_set.json trong V_real, cộng kết quả RL,
và các phân tích phụ.

Đầu ra (results/):
    benchmark_per_query.csv   tập chính (sự cố vị trí ngẫu nhiên) — mọi phương pháp
    benchmark_stress.csv      tập stress (sự cố ưu tiên trên tuyến dự kiến) — các phương pháp tạo chuyến đi
    saving_by_hour.csv        nhanh nhất vs ngắn nhất theo 24 giờ (V_base)
    noise_sweep.csv           độ nhạy theo σ (không sự cố)
    risk_eval.csv             độ tin cậy tuyến: plan-once vs P90 trên 100 kịch bản độc lập
    runtime_single.csv        thời gian tính ĐƠN LUỒNG (tuần tự, một tiến trình) — so sánh công bằng

Chạy: python -m src.evaluation.run_benchmark [--mode smoke|full] [--skip-rl] [--only main stress ...]
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.evaluation.parallel import pmap, state
from src.scenarios import load_split
from src.utils import derive_seed, setup_logging

log = logging.getLogger("benchmark")

TRIP_CLASSIC = ("dijkstra_length", "freeflow", "plan_once", "plan_once_sensor", "plan_once_gnn", "risk_p90",
                "replan", "replan_lazy", "replan_dstar", "replan_gnn", "oracle", "greedy_local")
SEARCH_ONLY = ("astar_length", "td_dijkstra", "td_astar_lb", "td_astar_alt", "td_astar_numba")


def _cached(key: str, builder: Callable[[], Any]) -> Any:
    st = state()
    if key not in st:
        st[key] = builder()
    return st[key]


def _tau_min() -> np.ndarray:
    from src.algorithms.heuristics import min_travel_time

    st = state()
    return _cached("tau_min", lambda: min_travel_time(st["graph"].length, st["V"]))


def _alt():
    from src.algorithms.heuristics import ALT

    st = state()
    return _cached("alt", lambda: ALT(st["graph"], _tau_min(), int(st["cfg"]["algorithms"]["alt"]["n_landmarks"]),
                                      seed=int(st["cfg"]["project"]["seed"])))


def _forecaster():
    from src.forecast import load_forecaster

    st = state()
    return _cached("forecaster", lambda: load_forecaster(st["cfg"], st["graph"], st["V"]))


def _bank():
    from src.algorithms.risk import ScenarioBank
    from src.speed_model import get_noise_model

    st = state()
    rc = st["cfg"]["algorithms"]["risk"]
    return _cached("bank", lambda: ScenarioBank(st["V"], st["graph"].length, get_noise_model(st["graph"], st["cfg"]),
                                                st["cfg"]["scenarios"]["sigma_levels"], int(rc["n_samples"]),
                                                derive_seed(st["cfg"]["project"]["seed"], "risk_bank")))


def _meta(q: dict[str, Any], cfg: Config, incident_key: str) -> dict[str, Any]:
    return {"query_id": q["id"], "hour": q["hour"], "depart_min": q["depart_min"], "sigma": q["sigma"],
            "n_incidents": len(q.get(incident_key, [])), "od_dist_m": q["od_dist_m"], "distance_bin": q["distance_bin"],
            "peak": q["hour"] in cfg["evaluation"]["peak_hours"]}


def _trip_row(method: str, tr: Any, meta: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {**meta, "method": method, "reached": tr.reached, "travel_time": tr.travel_time,
            "planned_time": tr.planned_time, "length_m": tr.length_m, "expanded": tr.expanded,
            "runtime_s": tr.runtime_s, "n_replans": tr.n_replans, "edges": " ".join(map(str, tr.edges)), **extra}


def _search_row(method: str, res: Any, meta: dict[str, Any], prep_s: float = 0.0) -> dict[str, Any]:
    return {**meta, "method": method, "reached": res.found, "travel_time": np.nan, "planned_time": res.cost,
            "length_m": np.nan, "expanded": res.expanded, "runtime_s": res.runtime_s, "n_replans": 0, "edges": "",
            "prep_s": prep_s}


def greedy_local(q: dict[str, Any], incident_key: str = "incidents") -> dict[str, Any]:
    """Baseline thủ công chỉ dùng quan sát CỤC BỘ như RL: chọn cạnh hướng về đích nhất, tránh nút đã thăm."""
    from src.env.routing_env import RoutingEnv

    st = state()
    env = _cached("greedy_env", lambda: RoutingEnv(st["graph"], st["V"], [q], st["cfg"], seed=0))
    t_start = time.perf_counter()
    obs, _ = env.reset(options={"query": q, "incident_key": incident_key})
    done, info = False, {"reached": False}
    while not done:
        f = obs[env.n_global:].reshape(-1, env.n_feat)[: env.K]
        score = f[:, 3] - 2.0 * f[:, 4]
        score[~env.action_masks()] = -np.inf
        obs, _, term, trunc, info = env.step(int(np.argmax(score)))
        done = term or trunc
    reached = bool(info["reached"])
    return {**_meta(q, st["cfg"], incident_key), "method": "greedy_local", "reached": reached,
            "travel_time": info["travel_time"] if reached else np.nan, "planned_time": np.nan,
            "length_m": env.graph.path_length(env.path_edges), "expanded": info["steps"],
            "runtime_s": time.perf_counter() - t_start, "n_replans": 0, "edges": " ".join(map(str, env.path_edges))}


def run_query(q: dict[str, Any], sigma: float | None = None, with_incidents: bool = True,
              methods: tuple[str, ...] | None = None, incident_key: str = "incidents") -> list[dict[str, Any]]:
    """Chạy các phương pháp cho một truy vấn (gọi trong worker)."""
    from src.algorithms.astar import astar_freeflow, astar_length
    from src.algorithms.dijkstra import shortest_by_length
    from src.algorithms.dstar_lite import replan_dstar
    from src.algorithms.heuristics import lb_heuristic
    from src.algorithms.numba_td import td_astar_numba, warmup
    from src.algorithms.replan import ObservedModel, oracle, plan_once, replan, static_trip, trip_from_plan
    from src.algorithms.risk import risk_route
    from src.algorithms.td_astar import td_astar, td_dijkstra
    from src.scenarios import build_real_model

    st = state()
    cfg, graph, forecast = st["cfg"], st["graph"], st["forecast"]
    ac = cfg["algorithms"]
    rp = ac["replan"]
    saf = float(ac["heuristic_safety"])
    real = build_real_model(graph, st["V"], q, cfg, sigma=sigma, with_incidents=with_incidents, incident_key=incident_key)
    s, t, t0 = q["origin"], q["dest"], q["depart_min"]
    meta = _meta(q, cfg, incident_key)
    if sigma is not None:
        meta["sigma"] = sigma
    want = set(methods or TRIP_CLASSIC + SEARCH_ONLY)
    rows = []
    if "dijkstra_length" in want:
        rows.append(_trip_row("dijkstra_length", static_trip("dijkstra_length", graph, shortest_by_length(graph, s, t), real, t0, s), meta))
    if "astar_length" in want:
        rows.append(_trip_row("astar_length", static_trip("astar_length", graph, astar_length(graph, s, t, saf), real, t0, s), meta))
    if "freeflow" in want:
        rows.append(_trip_row("freeflow", static_trip("freeflow", graph, astar_freeflow(graph, s, t, saf), real, t0, s), meta))
    if "td_dijkstra" in want:
        rows.append(_trip_row("td_dijkstra", trip_from_plan("td_dijkstra", graph, td_dijkstra(graph, s, t, t0, forecast), real, t0, s), meta))
    if "plan_once" in want:
        rows.append(_trip_row("plan_once", plan_once(graph, s, t, t0, forecast, real, saf), meta))
    if "td_astar_lb" in want:
        h, prep = lb_heuristic(graph, t, _tau_min())
        rows.append(_search_row("td_astar_lb", td_astar(graph, s, t, t0, forecast, heuristic=h), meta, prep))
    if "td_astar_alt" in want:
        alt = _alt()
        t1 = time.perf_counter()
        h = alt.heuristic(t)
        rows.append(_search_row("td_astar_alt", td_astar(graph, s, t, t0, forecast, heuristic=h), meta, time.perf_counter() - t1))
    if "td_astar_numba" in want:
        _cached("numba_warm", lambda: warmup(graph, forecast) or True)
        rows.append(_search_row("td_astar_numba", td_astar_numba(graph, s, t, t0, forecast), meta))
    fc = _forecaster() if want & {"plan_once_sensor", "plan_once_gnn", "replan_gnn"} else None
    if "plan_once_sensor" in want and fc is not None:
        t1 = time.perf_counter()
        idx = np.flatnonzero(fc.sensors)
        delays = dict(zip(idx.tolist(), (real.travel_time_vec(idx, t0) - forecast.travel_time_vec(idx, t0)).tolist()))
        plan = td_astar(graph, s, t, t0, ObservedModel(forecast, delays, forecast.v_max_mpm), safety=saf)
        tr = trip_from_plan("plan_once_sensor", graph, plan, real, t0, s)
        tr.runtime_s = time.perf_counter() - t1
        rows.append(_trip_row("plan_once_sensor", tr, meta))
    if "plan_once_gnn" in want and fc is not None:
        t1 = time.perf_counter()
        gm = fc.model_from_observations(real, t0, graph.edges_within(s, rp["observation_radius_m"]))
        plan = td_astar(graph, s, t, t0, _clamped(gm, forecast), safety=saf)
        tr = trip_from_plan("plan_once_gnn", graph, plan, real, t0, s)
        tr.runtime_s = time.perf_counter() - t1
        rows.append(_trip_row("plan_once_gnn", tr, meta))
    if "risk_p90" in want:
        rc = ac["risk"]
        tr = risk_route(graph, s, t, t0, forecast, real, _bank().bank(q["sigma"] if sigma is None else sigma),
                        int(rc["k_candidates"]), float(rc["quantile"]))
        rows.append(_trip_row("risk_p90", tr, meta, same_as_plan_once=tr.extra["same_as_plan_once"]))
    if "replan" in want:
        rows.append(_trip_row("replan", replan(graph, s, t, t0, forecast, real, rp["observation_radius_m"],
                                               rp["remember_observations"], saf), meta))
    if "replan_lazy" in want:
        rows.append(_trip_row("replan_lazy", replan(graph, s, t, t0, forecast, real, rp["observation_radius_m"],
                                                    rp["remember_observations"], saf, lazy_threshold=float(rp["lazy_threshold"]),
                                                    method="replan_lazy"), meta))
    if "replan_dstar" in want:
        tr = replan_dstar(graph, s, t, t0, forecast, real, rp["observation_radius_m"], float(ac["dstar"]["refresh_min"]))
        rows.append(_trip_row("replan_dstar", tr, meta, incremental_updates=tr.extra["incremental_updates"]))
    if "replan_gnn" in want and fc is not None:
        period = float(cfg["forecast"]["recompute_min"])
        cache: dict[str, Any] = {"t": -1e9, "m": None}

        def provider(u: int, tnow: float, _delays: dict[int, float]):
            # GCN chạy lại theo chu kỳ cập nhật cảm biến; quan sát cục bộ mới vẫn vào ngay qua δ (ObservedModel)
            if tnow - cache["t"] >= period:
                cache["t"], cache["m"] = tnow, fc.model_from_observations(
                    real, tnow, graph.edges_within(u, rp["observation_radius_m"]))
            return cache["m"]

        rows.append(_trip_row("replan_gnn", replan(graph, s, t, t0, forecast, real, rp["observation_radius_m"],
                                                   rp["remember_observations"], saf, base_provider=provider,
                                                   method="replan_gnn"), meta))
    if "oracle" in want:
        rows.append(_trip_row("oracle", oracle(graph, s, t, t0, real, saf), meta))
    if "greedy_local" in want:
        rows.append(greedy_local(q, incident_key))
    return rows


class _Clamped:
    """Mô hình dự báo GCN với cận dưới L/v_max của dự báo gốc (giữ heuristic TD-A* admissible)."""

    def __init__(self, model: Any, ref: Any) -> None:
        self.model, self.v_max_mpm, self._len = model, ref.v_max_mpm, ref._len

    def travel_time(self, e: int, t: float) -> float:
        return max(self.model.travel_time(e, t), self._len[e] / self.v_max_mpm)


def _clamped(model: Any, ref: Any) -> _Clamped:
    return _Clamped(model, ref)


def _stress_task(q: dict[str, Any]) -> list[dict[str, Any]]:
    return run_query(q, methods=TRIP_CLASSIC, incident_key="incidents_stress")


def _saving_task(q: dict[str, Any]) -> list[dict[str, Any]]:
    """Nhanh nhất (TD-A*) vs ngắn nhất (Dijkstra độ dài) trên V_base cho 24 giờ xuất phát."""
    from src.algorithms.dijkstra import shortest_by_length
    from src.algorithms.td_astar import td_astar

    st = state()
    graph, forecast = st["graph"], st["forecast"]
    short = shortest_by_length(graph, q["origin"], q["dest"])
    rows = []
    for h in range(24):
        t0 = 60.0 * h
        fast = td_astar(graph, q["origin"], q["dest"], t0, forecast)
        tt_short = forecast.path_travel_time(short.edges, t0)
        rows.append({"query_id": q["id"], "hour": h, "tt_fastest": fast.cost, "tt_shortest": tt_short,
                     "len_fastest_m": graph.path_length(fast.edges), "len_shortest_m": short.cost,
                     "saving_pct": 100 * (1 - fast.cost / tt_short)})
    return rows


def _noise_task(args: tuple[dict[str, Any], float]) -> list[dict[str, Any]]:
    q, sigma = args
    return run_query(q, sigma=sigma, with_incidents=False,
                     methods=("dijkstra_length", "plan_once", "plan_once_gnn", "replan", "replan_gnn", "oracle"))


def _risk_eval_task(q: dict[str, Any]) -> list[dict[str, Any]]:
    """Độ tin cậy: tuyến plan-once và tuyến P90 trên `eval_samples` kịch bản độc lập (không sự cố)."""
    from src.algorithms.risk import ScenarioBank, path_times, risk_route
    from src.algorithms.td_astar import td_astar
    from src.speed_model import get_noise_model

    st = state()
    cfg, graph, forecast = st["cfg"], st["graph"], st["forecast"]
    rc = cfg["algorithms"]["risk"]
    s, t, t0 = q["origin"], q["dest"], q["depart_min"]
    truth = ScenarioBank(st["V"], graph.length, get_noise_model(graph, cfg), [q["sigma"]], int(rc["eval_samples"]),
                         derive_seed(cfg["project"]["seed"], "risk_truth", q["id"])).bank(q["sigma"])
    po = td_astar(graph, s, t, t0, forecast).edges
    rk = risk_route(graph, s, t, t0, forecast, forecast, _bank().bank(q["sigma"]), int(rc["k_candidates"]),
                    float(rc["quantile"])).edges
    rows = []
    for name, edges in (("plan_once", po), ("risk_p90", rk)):
        tt = path_times(truth, graph.length, edges, t0)
        rows.append({"query_id": q["id"], "sigma": q["sigma"], "hour": q["hour"], "route": name, "mean": float(tt.mean()),
                     "p90": float(np.quantile(tt, 0.9)), "std": float(tt.std()), "same": po == rk})
    return rows


def runtime_single(cfg: Config, test: list[dict[str, Any]]) -> pd.DataFrame:
    """Thời gian tính tuần tự trong MỘT tiến trình, một luồng (sau khi làm nóng JIT/cache)."""
    import torch

    from src.evaluation.parallel import _init

    torch.set_num_threads(1)
    _init(cfg.mode)
    sub = test[: int(cfg["evaluation"]["single_thread_n_queries"])]
    methods = TRIP_CLASSIC + SEARCH_ONLY
    run_query(sub[0], methods=methods)  # làm nóng: ALT, GCN, ngân hàng kịch bản, Numba JIT
    rows = [r for q in sub for r in run_query(q, methods=methods)]
    df = pd.DataFrame(rows)
    out = df.groupby("method").agg(runtime_ms=("runtime_s", lambda s: 1000 * s.mean()),
                                   runtime_p90_ms=("runtime_s", lambda s: 1000 * s.quantile(0.9)),
                                   expanded=("expanded", "mean")).reset_index()
    if "prep_s" in df:
        prep = df.groupby("method").prep_s.mean().mul(1000).rename("prep_ms").reset_index()
        out = out.merge(prep, on="method", how="left")
    out = pd.concat([out, _rl_runtime_single(cfg, sub)], ignore_index=True)
    out["n_queries"] = len(sub)
    return out


def _rl_runtime_single(cfg: Config, sub: list[dict[str, Any]]) -> pd.DataFrame:
    from pathlib import Path

    from src.graph_data import RoadGraph
    from src.rl.evaluate_rl import load_model, rollout
    from src.rl.train import make_env
    from src.speed_model import load_vbase
    from src.utils import load_json

    p = cfg.path("results_dir") / "rl" / "train_summary.json"
    if not p.exists():
        return pd.DataFrame()
    summ = load_json(p)
    graph, V = RoadGraph.from_config(cfg), load_vbase(cfg)
    vmap = {v["name"]: v for v in cfg["rl"]["variants"]}
    rows = []
    for name, seed in summ["selected_seed"].items():
        run = next(r for r in summ["runs"] if r["variant"] == name and r["seed"] == seed)
        env = make_env(cfg, vmap[name], graph, V, sub, 0, eval_mode=True)
        model = load_model(Path(run["best_model_path"]), run["algo"])
        rollout(model, run["algo"], env, sub[0])
        rs = [rollout(model, run["algo"], env, q) for q in sub]
        rt = np.array([r["runtime_s"] for r in rs])
        rows.append({"method": name, "runtime_ms": 1000 * rt.mean(), "runtime_p90_ms": 1000 * np.quantile(rt, 0.9),
                     "expanded": float(np.mean([r["steps"] for r in rs]))})
    return pd.DataFrame(rows)


def merge_rl(cfg: Config, df: pd.DataFrame, test: list[dict[str, Any]], incident_set: str) -> pd.DataFrame:
    """Gộp kết quả RL (seed được chọn theo val) vào bảng benchmark."""
    path = cfg.path("results_dir") / "rl" / "rl_test_per_query.csv"
    if not path.exists():
        log.warning("Không có %s — bỏ qua RL trong benchmark", path)
        return df
    rl = pd.read_csv(path)
    rl = rl[rl.selected & (rl.incident_set == incident_set)]
    key = "incidents" if incident_set == "main" else "incidents_stress"
    meta = {q["id"]: _meta(q, cfg, key) for q in test}
    rows = [{**meta[r.query_id], "method": r.variant, "reached": bool(r.reached), "travel_time": r.travel_time,
             "planned_time": np.nan, "length_m": r.length_m, "expanded": r.steps, "runtime_s": r.runtime_s,
             "n_replans": 0, "edges": r.edges if isinstance(r.edges, str) else "", "mean_beta": r.mean_beta}
            for r in rl.itertuples()]
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


PARTS = ("main", "stress", "saving", "noise", "risk", "runtime")


def run(cfg: Config, include_rl: bool = True, only: list[str] | None = None) -> None:
    ec = cfg["evaluation"]
    nw = int(ec["n_workers"])
    test = load_split(cfg, "test")
    res = cfg.path("results_dir")
    parts = only or PARTS
    t0 = time.time()
    if "main" in parts:
        df = pd.DataFrame([r for rs in pmap(run_query, test, cfg.mode, nw) for r in rs])
        if include_rl:
            df = merge_rl(cfg, df, test, "main")
        df.to_csv(res / "benchmark_per_query.csv", index=False)
        log.info("Tập chính: %d truy vấn, %d dòng (%.0f s)", len(test), len(df), time.time() - t0)
    if "stress" in parts:
        ds = pd.DataFrame([r for rs in pmap(_stress_task, test, cfg.mode, nw) for r in rs])
        if include_rl:
            ds = merge_rl(cfg, ds, test, "stress")
        ds.to_csv(res / "benchmark_stress.csv", index=False)
        log.info("Tập stress xong (%.0f s)", time.time() - t0)
    if "saving" in parts:
        sub = test[: int(ec["saving_by_hour_n_queries"])]
        pd.DataFrame([r for rs in pmap(_saving_task, sub, cfg.mode, nw) for r in rs]).to_csv(res / "saving_by_hour.csv", index=False)
    if "noise" in parts:
        nq = test[: int(ec["noise_sweep_n_queries"])]
        tasks = [(q, float(s)) for s in ec["noise_sweep_sigmas"] for q in nq]
        pd.DataFrame([r for rs in pmap(_noise_task, tasks, cfg.mode, nw) for r in rs]).to_csv(res / "noise_sweep.csv", index=False)
        log.info("Quét nhiễu xong (%.0f s)", time.time() - t0)
    if "risk" in parts:
        rq = test[: int(cfg["algorithms"]["risk"]["eval_n_queries"])]
        pd.DataFrame([r for rs in pmap(_risk_eval_task, rq, cfg.mode, nw) for r in rs]).to_csv(res / "risk_eval.csv", index=False)
        log.info("Đánh giá độ tin cậy tuyến xong (%.0f s)", time.time() - t0)
    if "runtime" in parts:
        runtime_single(cfg, test).to_csv(res / "runtime_single.csv", index=False)
        log.info("Đo thời gian đơn luồng xong (%.0f s)", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--skip-rl", action="store_true")
    ap.add_argument("--only", nargs="*", choices=PARTS)
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg, include_rl=not args.skip_rl, only=args.only)


if __name__ == "__main__":
    main()
