"""Giai đoạn 6 — huấn luyện RL với curriculum 3 mức, nhiều seed, đánh giá định kỳ trên val.

Mỗi biến thể (configs/config.yaml → rl.variants) tự khai báo:
    algo        maskable_ppo | recurrent_ppo | dqn
    timesteps   ngân sách bước (bị chặn bởi rl.max_timesteps nếu có — dùng cho smoke)
    env_class   routing (chọn từng cạnh) | hybrid (RL chọn hệ số β, TD-A* tìm đường — mục B3)
    policy      mlp | tree_gnn (mục B1)
    bc          true → học trước bằng behavior cloning từ quỹ đạo oracle rồi tinh chỉnh PPO (mục B4)
    env         ghi đè cấu hình môi trường (obs_mode, reward.shaping, …)
Mỗi (biến thể, seed) chạy trong một tiến trình riêng. Log TensorBoard ở results/rl/tb.

Chạy: python -m src.rl.train [--mode smoke|full] [--variants rl_tree rl_lstm]
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
from src.utils import save_json, set_global_seed, setup_logging

log = logging.getLogger("train")


def variant_timesteps(cfg: Config, variant: dict[str, Any]) -> int:
    n = int(variant.get("timesteps", cfg["rl"]["total_timesteps"]))
    cap = cfg["rl"].get("max_timesteps")
    return min(n, int(cap)) if cap else n


def make_env(cfg: Config, variant: dict[str, Any], graph: Any, V_base: np.ndarray, queries: list[dict[str, Any]],
             seed: int, eval_mode: bool = False) -> Any:
    from src.env.hybrid_env import HybridEnv
    from src.env.routing_env import RoutingEnv

    cls = HybridEnv if variant.get("env_class", "routing") == "hybrid" else RoutingEnv
    return cls(graph, V_base, queries, cfg, variant.get("env"), level=3, seed=seed)


def _build_model(cfg: Config, variant: dict[str, Any], env: Any, seed: int, tb_dir: Path) -> Any:
    rc = cfg["rl"]
    algo = variant.get("algo", rc["algo"])
    from src.config import deep_merge

    p = deep_merge(rc["ppo"], variant.get("ppo") or {})
    pk: dict[str, Any] = {"net_arch": list(p["net_arch"])}
    if variant.get("policy") == "tree_gnn":
        from src.rl.policies import TreeGNNExtractor

        e0 = env.envs[0] if hasattr(env, "envs") else env
        inner = getattr(e0, "unwrapped", e0)
        pk.update(features_extractor_class=TreeGNNExtractor,
                  features_extractor_kwargs={"n_global": inner.n_global, "K": inner.K, "depth": inner.depth,
                                             "n_feat": inner.n_feat, "hidden": int(rc["gnn_hidden"])})
    common = dict(learning_rate=p["learning_rate"], n_steps=p["n_steps"], batch_size=p["batch_size"],
                  n_epochs=p["n_epochs"], gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
                  ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], policy_kwargs=pk, tensorboard_log=str(tb_dir),
                  seed=seed, device="cpu", verbose=0)
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO("MlpPolicy", env, **common)
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        pk.update(lstm_hidden_size=int(rc["lstm_hidden"]), n_lstm_layers=1)
        return RecurrentPPO("MlpLstmPolicy", env, **common)
    if algo == "dqn":
        from stable_baselines3 import DQN

        q = rc["dqn"]
        return DQN("MlpPolicy", env, learning_rate=q["learning_rate"], buffer_size=q["buffer_size"],
                   learning_starts=q["learning_starts"], batch_size=q["batch_size"], gamma=q["gamma"],
                   exploration_fraction=q["exploration_fraction"], exploration_final_eps=q["exploration_final_eps"],
                   target_update_interval=q["target_update_interval"], policy_kwargs={"net_arch": list(q["net_arch"])},
                   tensorboard_log=str(tb_dir), seed=seed, device="cpu", verbose=0)
    raise ValueError(f"algo không hợp lệ: {algo}")


def behavior_cloning(model: Any, env: Any, cfg: Config, graph: Any, seed: int) -> dict[str, float]:
    """Mục B4: thu quỹ đạo có nhãn từ oracle (TD-A* trên V_real của tập), DAgger nhẹ (trộn hành động ngẫu nhiên,
    nhãn luôn là cạnh tiếp theo của oracle từ trạng thái hiện tại), rồi cực đại log-likelihood của chính sách."""
    import torch

    from src.algorithms.td_astar import td_astar

    bc = cfg["rl"]["bc"]
    rng = np.random.default_rng(seed)
    obs_l, act_l, mask_l = [], [], []
    t0 = time.time()
    for _ in range(int(bc["n_episodes"])):
        obs, _ = env.reset()
        plan: list[int] = td_astar(graph, env.u, env.dest, env.t, env.real).edges
        done = False
        while not done and plan:
            out = graph.out_of(env.u)
            a_star = int(np.flatnonzero(out == plan[0])[0])
            mask = env.action_masks()
            obs_l.append(obs)
            act_l.append(a_star)
            mask_l.append(mask)
            if rng.random() < bc["explore_prob"]:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                a = a_star
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            if a == a_star:
                plan = plan[1:]
            elif not done:
                plan = td_astar(graph, env.u, env.dest, env.t, env.real).edges
    X = torch.as_tensor(np.asarray(obs_l), dtype=torch.float32)
    A = torch.as_tensor(np.asarray(act_l), dtype=torch.long)
    M = np.asarray(mask_l)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=float(bc["lr"]))
    n = len(X)
    losses = []
    for _ in range(int(bc["epochs"])):
        perm = rng.permutation(n)
        for i in range(0, n, int(bc["batch_size"])):
            b = perm[i: i + int(bc["batch_size"])]
            _, logp, _ = pol.evaluate_actions(X[b], A[b], action_masks=M[b])
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    with torch.no_grad():
        dist = pol.get_distribution(X[:5000], action_masks=M[:5000])
        acc = float((dist.distribution.probs.argmax(dim=1) == A[:5000]).float().mean())
    log.info("BC: %d mẫu, %d epoch, loss cuối %.3f, độ chính xác bắt chước %.1f%% (%.0f s)", n, bc["epochs"],
             losses[-1], 100 * acc, time.time() - t0)
    return {"bc_samples": n, "bc_final_loss": losses[-1], "bc_accuracy": acc, "bc_seconds": time.time() - t0}


def train_one(mode: str, variant: dict[str, Any], seed: int) -> dict[str, Any]:
    """Huấn luyện một (biến thể, seed). Hàm cấp module để chạy được trong ProcessPoolExecutor (spawn)."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    from src.algorithms.replan import oracle
    from src.graph_data import RoadGraph
    from src.rl.evaluate_rl import evaluate_policy
    from src.scenarios import build_real_model, load_split
    from src.speed_model import load_vbase
    from src.utils import derive_seed

    torch.set_num_threads(1)
    cfg = load_config(mode)
    setup_logging(cfg["project"]["log_level"])
    rc = cfg["rl"]
    name = variant["name"]
    algo = variant.get("algo", rc["algo"])
    run_seed = int(cfg["project"]["seed"]) * 1000 + seed
    set_global_seed(run_seed)
    graph = RoadGraph.from_config(cfg)
    V_base = load_vbase(cfg)
    train_q = load_split(cfg, "train")
    val_q = load_split(cfg, "val")[: int(rc["eval_n_queries"])]
    run_dir = cfg.path("models_dir") / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = cfg.path("results_dir") / "rl" / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    n_envs = int(variant.get("n_envs", rc["n_envs"]))

    def factory(r: int):
        return lambda: make_env(cfg, variant, graph, V_base, train_q, derive_seed(run_seed, "env", r))

    venv = VecMonitor(DummyVecEnv([factory(r) for r in range(n_envs)]), filename=str(run_dir / "monitor"))
    venv.env_method("set_level", 1)
    eval_env = make_env(cfg, variant, graph, V_base, val_q, run_seed + 1, eval_mode=True)
    oracle_tt = np.array([oracle(graph, q["origin"], q["dest"], q["depart_min"],
                                 build_real_model(graph, V_base, q, cfg)).travel_time for q in val_q])
    total = variant_timesteps(cfg, variant)
    eval_freq = int(variant.get("eval_freq", rc["eval_freq"]))
    fr = rc["curriculum"]["fractions"]
    history: list[dict[str, Any]] = []
    t_start = time.time()
    model = _build_model(cfg, variant, venv, run_seed, tb_dir)
    bc_info: dict[str, Any] = {}
    if variant.get("bc"):
        bc_env = make_env(cfg, variant, graph, V_base, train_q, run_seed + 7)
        bc_env.set_level(3)
        bc_info = behavior_cloning(model, bc_env, cfg, graph, run_seed)

    class CurriculumEvalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.level = 1
            self.next_eval = eval_freq
            self.best_score = -np.inf

        def _on_training_start(self) -> None:
            if variant.get("bc"):
                self._evaluate(0)  # điểm xuất phát sau BC

        def _on_step(self) -> bool:
            n = self.num_timesteps
            level = 1 if n < fr[0] * total else 2 if n < fr[1] * total else 3
            if level != self.level:
                self.level = level
                self.training_env.env_method("set_level", level)
                log.info("[%s s%d] curriculum -> mức %d tại %d bước", name, seed, level, n)
            if n >= self.next_eval:
                self.next_eval += eval_freq
                self._evaluate(n)
            return True

        def _evaluate(self, n: int) -> None:
            df = evaluate_policy(self.model, algo, eval_env, val_q)
            ok = df.reached.to_numpy()
            regret = 100 * (df.travel_time.to_numpy() / oracle_tt - 1)
            mean_regret = float(np.nanmean(regret[ok])) if ok.any() else float("nan")
            success = float(ok.mean())
            score = success - 0.002 * (mean_regret if ok.any() else 1000.0)
            self.logger.record("val/success_rate", success)
            self.logger.record("val/regret_pct", mean_regret)
            self.logger.record("val/score", score)
            self.logger.record("train/curriculum_level", self.level)
            history.append({"timesteps": n, "success_rate": success, "regret_pct": mean_regret,
                            "score": score, "level": self.level, "wall_s": time.time() - t_start})
            if score > self.best_score:
                self.best_score = score
                self.model.save(run_dir / "best_model")
            log.info("[%s s%d] %7d bước | val tới đích %.1f%% | regret %.1f%% | mức %d", name, seed, n,
                     100 * success, mean_regret, self.level)

    cb = CurriculumEvalCallback()
    model.learn(total_timesteps=total, callback=cb, tb_log_name=f"{name}_seed{seed}", progress_bar=False)
    if not history or history[-1]["timesteps"] < model.num_timesteps:
        cb._evaluate(model.num_timesteps)
    model.save(run_dir / "final_model")
    pd.DataFrame(history).to_csv(run_dir / "eval_history.csv", index=False)
    best = max(history, key=lambda h: h["score"])
    return {"variant": name, "algo": algo, "seed": seed, "best_score": best["score"],
            "best_val_success": best["success_rate"], "best_val_regret_pct": best["regret_pct"],
            "best_at_timesteps": best["timesteps"], "final_val_success": history[-1]["success_rate"],
            "final_val_regret_pct": history[-1]["regret_pct"], "train_wall_s": time.time() - t_start,
            "timesteps": int(model.num_timesteps), "best_model_path": str(run_dir / "best_model.zip"),
            "run_dir": str(run_dir), **bc_info}


def run(cfg: Config, only: list[str] | None = None) -> dict[str, Any]:
    rc = cfg["rl"]
    variants = [v for v in rc["variants"] if not only or v["name"] in only]
    tasks = [(cfg.mode, v, int(s)) for v in variants for s in rc["seeds"]]
    # lượt dài chạy trước để tận dụng song song
    tasks.sort(key=lambda t: -variant_timesteps(cfg, t[1]))
    log.info("Huấn luyện %d lượt trên %d tiến trình: %s", len(tasks), rc["n_parallel_runs"],
             ", ".join(f"{v['name']}({variant_timesteps(cfg, v)})" for v in variants))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(int(rc["n_parallel_runs"]), len(tasks))) as ex:
        runs = list(ex.map(train_one, *zip(*tasks)))
    summary_path = cfg.path("results_dir") / "rl" / "train_summary.json"
    if only and summary_path.exists():  # giữ kết quả các biến thể không huấn luyện lại
        from src.utils import load_json

        old = load_json(summary_path)
        runs = [r for r in old["runs"] if r["variant"] not in only] + runs
    selected = {}
    for v in rc["variants"]:
        cand = [r for r in runs if r["variant"] == v["name"]]
        if cand:
            selected[v["name"]] = max(cand, key=lambda r: r["best_score"])["seed"]
    summary = {"runs": runs, "selected_seed": selected, "wall_s": time.time() - t0}
    save_json(summary, summary_path)
    for r in sorted(runs, key=lambda r: (r["variant"], r["seed"])):
        log.info("%-12s seed %d: val tốt nhất tới đích %.1f%%, regret %.1f%% (tại %d bước); %.0f s",
                 r["variant"], r["seed"], 100 * r["best_val_success"], r["best_val_regret_pct"],
                 r["best_at_timesteps"], r["train_wall_s"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--variants", nargs="*", help="chỉ huấn luyện các biến thể này")
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg, args.variants)


if __name__ == "__main__":
    main()
