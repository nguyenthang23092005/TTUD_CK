"""Chạy toàn bộ pipeline theo thứ tự.

    python run_all.py --mode smoke            # kiểm tra nhanh (vài phút)
    python run_all.py --mode full             # chạy đầy đủ
    python run_all.py --mode full --skip-rl   # bỏ huấn luyện/đánh giá RL
    python run_all.py --mode full --steps benchmark plots   # chỉ chạy một số bước

Các bước: graph, speeds, scenarios, forecast, train_rl, eval_rl, departure, benchmark, multiagent, plots, tests
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

from src.config import load_config
from src.utils import save_json, set_global_seed, setup_logging

STEPS = ["graph", "speeds", "scenarios", "forecast", "train_rl", "eval_rl", "departure", "benchmark", "multiagent",
         "plots", "tests"]
RL_STEPS = {"train_rl", "eval_rl"}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252 — cần trước cả --help
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--skip-rl", action="store_true", help="bỏ qua huấn luyện và đánh giá RL")
    ap.add_argument("--force-download", action="store_true", help="tải lại dữ liệu OSM (bỏ qua cache)")
    ap.add_argument("--steps", nargs="*", choices=STEPS, help="chỉ chạy các bước này")
    args = ap.parse_args()

    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"], cfg.path("results_dir") / "logs" / "run_all.log")
    log = logging.getLogger("run_all")
    set_global_seed(int(cfg["project"]["seed"]))
    steps = args.steps or STEPS
    if args.skip_rl:
        steps = [s for s in steps if s not in RL_STEPS]
    timings: dict[str, float] = {}
    log.info("=== Pipeline mode=%s, các bước: %s ===", args.mode, steps)
    for step in steps:
        t0 = time.time()
        log.info(">>> Bước %s", step)
        if step == "graph":
            from src import graph_builder

            graph_builder.run(cfg, force_download=args.force_download)
        elif step == "speeds":
            from src import speed_model

            speed_model.run(cfg)
        elif step == "scenarios":
            from src import scenarios

            scenarios.run(cfg)
        elif step == "forecast":
            from src import forecast

            forecast.train(cfg)
        elif step == "multiagent":
            from src.evaluation import multiagent

            multiagent.run(cfg)
        elif step == "train_rl":
            from src.rl import train

            train.run(cfg)
        elif step == "eval_rl":
            from src.rl import evaluate_rl

            evaluate_rl.run(cfg)
        elif step == "departure":
            from src.rl import departure_time

            departure_time.run(cfg)
        elif step == "benchmark":
            from src.evaluation import run_benchmark

            run_benchmark.run(cfg, include_rl=not args.skip_rl)
        elif step == "plots":
            from src.evaluation import plots

            plots.run(cfg)
        elif step == "tests":
            rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"])
            if rc != 0:
                log.error("pytest thất bại (mã %d)", rc)
                sys.exit(rc)
        timings[step] = time.time() - t0
        log.info("<<< Xong %s trong %.1f s", step, timings[step])
    save_json(timings, cfg.path("results_dir") / "logs" / "timings.json")
    log.info("=== Hoàn tất pipeline (%.1f phút) ===", sum(timings.values()) / 60)


if __name__ == "__main__":
    main()
