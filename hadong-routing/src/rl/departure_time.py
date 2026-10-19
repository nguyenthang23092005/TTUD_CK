"""Thí nghiệm phụ — chọn giờ xuất phát: đi ngay hay đợi thêm `wait_step` phút (tối đa `max_waits` lần).

Chi phí = w · thời_gian_chờ + thời_gian_đi (TD-A* oracle trên V_real của truy vấn). w = 1 đúng định nghĩa
"tổng thời gian chờ + di chuyển"; w < 1 (độ nhạy) mô tả trường hợp chờ ở nhà/cơ quan ít tốn kém hơn ngồi trên xe.

Q-learning dạng bảng. Trạng thái = (khung 30 phút trong ngày, số lần đã chờ, nhóm khoảng cách O–D, nhóm mức ùn
DỰ BÁO của chính cặp O–D tại thời điểm quyết định = TT_dự_báo(t)/TT_tự_do) — đặc trưng riêng theo O–D (mục E1),
thông tin có sẵn lúc quyết định (không dùng V_real). So sánh với "đi ngay" và "giờ tối ưu vét cạn" (oracle).

Chạy: python -m src.rl.departure_time [--mode smoke|full]
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.evaluation.parallel import pmap, state
from src.scenarios import load_split
from src.utils import derive_seed, save_json, setup_logging

log = logging.getLogger("departure_time")
SLOT_MIN = 30.0


def _tt_row(q: dict[str, Any]) -> tuple[list[float], list[float], float]:
    """(thời gian oracle thực, thời gian dự báo) khi xuất phát lúc depart + k·wait_step, và thời gian tự do."""
    from src.algorithms.astar import astar_freeflow
    from src.algorithms.td_astar import td_astar
    from src.scenarios import build_real_model

    st = state()
    cfg, graph, fc = st["cfg"], st["graph"], st["forecast"]
    dc = cfg["departure"]
    real = build_real_model(graph, st["V"], q, cfg)
    ts = [q["depart_min"] + k * dc["wait_step_min"] for k in range(int(dc["max_waits"]) + 1)]
    tt_real = [td_astar(graph, q["origin"], q["dest"], t, real).cost for t in ts]
    tt_fc = [td_astar(graph, q["origin"], q["dest"], t, fc).cost for t in ts]
    return tt_real, tt_fc, astar_freeflow(graph, q["origin"], q["dest"]).cost


def _slot(t: float) -> int:
    return int((t % 1440.0) // SLOT_MIN) % int(1440 // SLOT_MIN)


def _cbin(ratio: float, edges: list[float]) -> int:
    return int(np.searchsorted(edges, ratio, side="right"))


def train_q(queries: list[dict[str, Any]], TT: np.ndarray, CB: np.ndarray, cfg: Config, w: float,
            seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Q[slot, j, dbin, cbin, a], a=0 đi, a=1 chờ. Reward = −chi phí. CB[i, k] = nhóm mức ùn dự báo.

    Trả thêm N (số lần cập nhật mỗi ô). Vì Q khởi tạo 0 trong khi mọi phần thưởng âm, ô ít được ghé trông "lạc quan";
    khi ra quyết định ta chỉ tin Q ở ô đã được cập nhật đủ `min_visits` lần (xem q_policy_k)."""
    dc = cfg["departure"]
    step, W = float(dc["wait_step_min"]), int(dc["max_waits"])
    n_bins = len(cfg["scenarios"]["distance_bins_m"]) - 1
    n_c = len(dc["congestion_bins"]) + 1
    Q = np.zeros((int(1440 // SLOT_MIN), W + 1, n_bins, n_c, 2))
    Q[:, W, :, :, 1] = -np.inf  # không được chờ quá max_waits
    N = np.zeros_like(Q)
    rng = np.random.default_rng(seed)
    n_ep = int(dc["episodes"])
    for ep in range(n_ep):
        eps = dc["epsilon_start"] + (dc["epsilon_end"] - dc["epsilon_start"]) * ep / max(n_ep - 1, 1)
        i = int(rng.integers(len(queries)))
        q, b = queries[i], queries[i]["distance_bin"]
        t, j = q["depart_min"], 0
        while True:
            s = (_slot(t), j, b, int(CB[i, j]))
            a = int(rng.integers(2)) if (j < W and rng.random() < eps) else int(np.argmax(Q[s]))
            if a == 0:
                Q[s + (0,)] += dc["alpha"] * (-TT[i, j] - Q[s + (0,)])
                N[s + (0,)] += 1
                break
            t2, j2 = t + step, j + 1
            s2 = (_slot(t2), j2, b, int(CB[i, j2]))
            Q[s + (1,)] += dc["alpha"] * (-w * step + dc["gamma"] * np.max(Q[s2]) - Q[s + (1,)])
            N[s + (1,)] += 1
            t, j = t2, j2
    return Q, N


def q_policy_k(Q: np.ndarray, N: np.ndarray, q: dict[str, Any], cb_row: np.ndarray, cfg: Config) -> int:
    """Số lần chờ k mà chính sách tham lam theo Q chọn; ô chưa học đủ (N < min_visits) → mặc định đi ngay."""
    dc = cfg["departure"]
    step, W = float(dc["wait_step_min"]), int(dc["max_waits"])
    t, j = q["depart_min"], 0
    while j < W:
        s = (_slot(t), j, q["distance_bin"], int(cb_row[j]))
        if N[s].min() < dc["min_visits"] or np.argmax(Q[s]) != 1:
            break
        t += step
        j += 1
    return j


def _unpack(rows: list[tuple[list[float], list[float], float]], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    TT = np.array([r[0] for r in rows])
    FC = np.array([r[1] for r in rows])
    FF = np.array([r[2] for r in rows])
    CB = np.vectorize(lambda x: _cbin(x, cfg["departure"]["congestion_bins"]))(FC / FF[:, None])
    return TT, CB


def run(cfg: Config) -> dict[str, Any]:
    dc = cfg["departure"]
    n_workers = int(cfg["evaluation"]["n_workers"])
    train = load_split(cfg, "train")[: int(dc["n_train_queries"])]
    test = load_split(cfg, "test")
    log.info("Tính bảng thời gian (oracle thực + dự báo) cho %d train + %d test truy vấn × %d giờ xuất phát",
             len(train), len(test), dc["max_waits"] + 1)
    TT_tr, CB_tr = _unpack(pmap(_tt_row, train, cfg.mode, n_workers), cfg)
    TT_te, CB_te = _unpack(pmap(_tt_row, test, cfg.mode, n_workers), cfg)
    step = float(dc["wait_step_min"])
    ks = np.arange(int(dc["max_waits"]) + 1)
    out_dir = cfg.path("results_dir") / "departure"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in dc["wait_cost_weights"]:
        Q, N = train_q(train, TT_tr, CB_tr, cfg, float(w), derive_seed(cfg["project"]["seed"], "departure", str(w)))
        np.save(out_dir / f"Q_w{w}.npy", Q)
        for i, q in enumerate(test):
            cost = w * step * ks + TT_te[i]
            k_q = q_policy_k(Q, N, q, CB_te[i], cfg)
            k_opt = int(np.argmin(cost))
            for pol, k in (("go_now", 0), ("q_learning", k_q), ("oracle_best", k_opt)):
                rows.append({"wait_weight": w, "policy": pol, "query_id": q["id"], "hour": q["hour"], "k_wait": k,
                             "wait_min": k * step, "travel_min": TT_te[i, k], "total_min": k * step + TT_te[i, k],
                             "cost": cost[k], "regret_cost": cost[k] - cost[k_opt]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "departure_results.csv", index=False)
    agg = (df.groupby(["wait_weight", "policy"])
           .agg(cost_mean=("cost", "mean"), travel_mean=("travel_min", "mean"), total_mean=("total_min", "mean"),
                wait_share=("k_wait", lambda s: float((s > 0).mean())), regret_mean=("regret_cost", "mean"))
           .reset_index())
    agg.to_csv(out_dir / "departure_summary.csv", index=False)
    summary = {"summary": agg.to_dict(orient="records"), "tt_profile_test_mean": TT_te.mean(axis=0).tolist(),
               "share_where_waiting_helps_w1": float((np.argmin(step * ks + TT_te, axis=1) > 0).mean()),
               "max_saving_by_waiting_min": float((TT_te[:, 0] - TT_te.min(axis=1)).max())}
    save_json(summary, out_dir / "departure_summary.json")
    for r in agg.itertuples():
        log.info("w=%.2f %-12s chi phí TB %.2f | đi TB %.2f | tỉ lệ chờ %.1f%% | regret %.3f", r.wait_weight,
                 r.policy, r.cost_mean, r.travel_mean, 100 * r.wait_share, r.regret_mean)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg)


if __name__ == "__main__":
    main()
