"""Chạy chính sách RL (deterministic) trên tập test trong V_real và ghi kết quả từng truy vấn.

Đánh giá mọi (biến thể × seed) trên hai bộ sự cố: tập chính (vị trí ngẫu nhiên) và tập stress (ưu tiên trên tuyến).

Chạy: python -m src.rl.evaluate_rl [--mode smoke|full]
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.utils import load_json, save_json, setup_logging

log = logging.getLogger("evaluate_rl")
INCIDENT_SETS = {"main": "incidents", "stress": "incidents_stress"}


def load_model(path: Path, algo: str) -> Any:
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path, device="cpu")
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(path, device="cpu")
    from stable_baselines3 import DQN

    return DQN.load(path, device="cpu")


def policy_action(model: Any, obs: np.ndarray, mask: np.ndarray, algo: str, state: Any = None,
                  episode_start: bool = False) -> tuple[int, Any]:
    """Hành động tất định. MaskablePPO dùng mask; DQN chọn argmax Q trên hành động hợp lệ;
    RecurrentPPO KHÔNG hỗ trợ mask (sb3-contrib) — hành động không hợp lệ bị môi trường phạt."""
    if algo == "maskable_ppo":
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(a), None
    if algo == "recurrent_ppo":
        a, state = model.predict(obs, state=state, episode_start=np.array([episode_start]), deterministic=True)
        return int(a), state
    import torch

    with torch.no_grad():
        q = model.q_net(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]
    q[~mask] = -np.inf
    return int(np.argmax(q)), None


def rollout(model: Any, algo: str, env: Any, query: dict[str, Any], incident_key: str = "incidents") -> dict[str, Any]:
    """Một tập đánh giá trên đúng truy vấn `query` (V_real tái lập theo seed của truy vấn)."""
    t_start = time.perf_counter()
    obs, _ = env.reset(options={"query": query, "incident_key": incident_key})
    done, info, total_r = False, {"reached": False}, 0.0
    state, start = None, True
    betas: list[float] = []
    while not done:
        a, state = policy_action(model, obs, env.action_masks(), algo, state, start)
        start = False
        obs, r, term, trunc, info = env.step(a)
        if "beta" in info:
            betas.append(info["beta"])
        total_r += r
        done = term or trunc
    reached = bool(info["reached"])
    edges = list(env.path_edges)
    graph = env.graph if hasattr(env, "graph") else env.inner.graph
    return {"query_id": query["id"], "reached": reached,
            "travel_time": float(info["travel_time"]) if reached else float("nan"),
            "length_m": graph.path_length(edges), "steps": int(info["steps"]),
            "revisits": int(info["steps"] - len(set(edges))), "reward": total_r,
            "runtime_s": time.perf_counter() - t_start, "edges": edges,
            "mean_beta": float(np.mean(betas)) if betas else float("nan")}


def evaluate_policy(model: Any, algo: str, env: Any, queries: list[dict[str, Any]],
                    incident_key: str = "incidents") -> pd.DataFrame:
    return pd.DataFrame([rollout(model, algo, env, q, incident_key) for q in queries])


def _eval_run(args: tuple[str, dict[str, Any], dict[str, Any], str]) -> pd.DataFrame:
    mode, variant, run_info, selected = args
    import torch

    from src.graph_data import RoadGraph
    from src.rl.train import make_env
    from src.scenarios import load_split
    from src.speed_model import load_vbase

    torch.set_num_threads(1)
    cfg = load_config(mode)
    graph = RoadGraph.from_config(cfg)
    V_base = load_vbase(cfg)
    test = load_split(cfg, "test")
    env = make_env(cfg, variant, graph, V_base, test, 0, eval_mode=True)
    model = load_model(Path(run_info["best_model_path"]), run_info["algo"])
    frames = []
    for set_name, key in INCIDENT_SETS.items():
        df = evaluate_policy(model, run_info["algo"], env, test, key)
        df["incident_set"] = set_name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["variant"], out["seed"], out["selected"] = variant["name"], run_info["seed"], run_info["seed"] == selected
    return out


def run(cfg: Config) -> pd.DataFrame:
    """Đánh giá mọi biến thể × seed trên test; đánh dấu seed được chọn (tốt nhất theo val)."""
    rl_dir = cfg.path("results_dir") / "rl"
    summary_path = rl_dir / "train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Chưa có {summary_path} — chạy `python -m src.rl.train --mode {cfg.mode}`")
    summary = load_json(summary_path)
    vmap = {v["name"]: v for v in cfg["rl"]["variants"]}
    tasks = [(cfg.mode, vmap[r["variant"]], r, summary["selected_seed"].get(r["variant"]))
             for r in summary["runs"] if r["variant"] in vmap]
    with ProcessPoolExecutor(max_workers=min(int(cfg["rl"]["n_parallel_runs"]), len(tasks))) as ex:
        frames = list(ex.map(_eval_run, tasks))
    out = pd.concat(frames, ignore_index=True)
    out["edges"] = out["edges"].map(lambda es: " ".join(map(str, es)))
    out.to_csv(rl_dir / "rl_test_per_query.csv", index=False)
    agg = (out.groupby(["incident_set", "variant", "seed"])
           .agg(success=("reached", "mean"), tt_mean=("travel_time", "mean"), steps=("steps", "mean"),
                runtime_ms=("runtime_s", lambda s: 1000 * s.mean())).reset_index())
    save_json(agg.to_dict(orient="records"), rl_dir / "rl_test_summary.json")
    for r in agg[agg.incident_set == "main"].itertuples():
        log.info("%-12s seed %d: tới đích %.1f%%, TB (khi tới) %.2f phút", r.variant, r.seed, 100 * r.success, r.tt_mean)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg)


if __name__ == "__main__":
    main()
