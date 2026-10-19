"""Dự báo ngắn hạn vận tốc bằng mạng tích chập đồ thị (GCN) — mục D1.

Bài toán: tại thời điểm t, ta chỉ quan sát được tỉ số vận tốc thực/dự báo r_e = log(v_real/v_base) trên một tập con
cạnh (cảm biến cố định ~15% cạnh, ưu tiên trục chính, cộng các cạnh trong bán kính quan sát quanh xe). Cần ước lượng
r cho MỌI cạnh tại hai mốc giờ h=⌊t/60⌋ và h+1 để lập kế hoạch TD-A*. Vì nhiễu môi trường có tương quan không gian–
thời gian (speed_model.NoiseModel), thông tin từ cạnh quan sát lan được sang cạnh lân cận.

Mô hình: X (E×F) → Linear → L lớp [Â·H·W₁ + H·W₂, ReLU] → Linear → (E×2), với Â là ma trận kề cạnh–cạnh chuẩn hoá.
Đặc trưng: r quan sát (0 nếu không), cờ quan sát, one-hot nhóm đường, sin/cos giờ, mức ùn nền c(e,h).
Huấn luyện có giám sát trên các "ngày" mô phỏng (seed độc lập với tập test), loss MSE trên mọi cạnh.

Chạy: python -m src.forecast [--mode smoke|full]   (huấn luyện + đánh giá, lưu results/forecast/)
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, load_config
from src.graph_data import RoadGraph
from src.speed_model import TravelTimeModel, base_model, edge_adjacency, get_noise_model, load_vbase
from src.utils import derive_seed, save_json, setup_logging

log = logging.getLogger("forecast")


def sensor_mask(graph: RoadGraph, cfg: Config) -> np.ndarray:
    """Tập cảm biến cố định: mọi cạnh thuộc nhóm `sensor_groups_always` + chọn ngẫu nhiên tới đủ tỉ lệ."""
    fc = cfg["forecast"]
    rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "sensors"))
    gid = [graph.group_names.index(g) for g in fc["sensor_groups_always"]]
    m = np.isin(graph.group_id, gid)
    need = int(fc["sensor_fraction"] * graph.n_edges) - int(m.sum())
    if need > 0:
        m[rng.choice(np.flatnonzero(~m), size=need, replace=False)] = True
    return m


def _torch_adj(graph: RoadGraph, cfg: Config):
    import torch

    A = edge_adjacency(graph, cfg["speed"]["spatial_noise"]["self_weight"]).tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A.data, dtype=torch.float32), A.shape).coalesce()


def build_gcn(n_in: int, hidden: int, layers: int):
    import torch
    from torch import nn

    class GCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(n_in, hidden)
            self.w_nb = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
            self.w_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.out = nn.Linear(hidden, 2)

        def forward(self, X: "torch.Tensor", A: "torch.Tensor") -> "torch.Tensor":
            H = torch.relu(self.inp(X))
            for wn, ws in zip(self.w_nb, self.w_self):
                H = torch.relu(torch.sparse.mm(A, wn(H)) + ws(H))
            return self.out(H)

    return GCN()


class Forecaster:
    """Bao gói GCN đã huấn luyện + hàm dựng đặc trưng + dựng mô hình thời gian dự báo cho TD-A*."""

    N_FEAT = 1 + 1 + 5 + 2 + 1

    def __init__(self, graph: RoadGraph, V_base: np.ndarray, cfg: Config, model: Any | None = None) -> None:
        import torch

        torch.set_num_threads(1)
        self.graph, self.V_base, self.cfg = graph, V_base, cfg
        self.fc = cfg["forecast"]
        self.A = _torch_adj(graph, cfg)
        self.onehot = np.eye(5)[graph.group_id]
        self.cong = 1.0 - V_base / graph.v_free[:, None]  # mức ùn nền (E,24)
        self.sensors = sensor_mask(graph, cfg)
        self.rho = float(cfg["speed"]["real_noise"]["rho_hourly"])
        self.model = model or build_gcn(self.N_FEAT, self.fc["gcn"]["hidden"], self.fc["gcn"]["layers"])
        self.base = base_model(graph, V_base, cfg)

    # ------------------------------------------------------------------
    def features(self, obs_lr: np.ndarray, obs_mask: np.ndarray, t: float) -> np.ndarray:
        h = int((t % 1440.0) // 60.0)
        ang = 2 * math.pi * (t % 1440.0) / 1440.0
        E = self.graph.n_edges
        return np.column_stack([np.where(obs_mask, obs_lr, 0.0), obs_mask.astype(float), self.onehot,
                                np.full(E, math.sin(ang)), np.full(E, math.cos(ang)), self.cong[:, h]]).astype(np.float32)

    def predict(self, obs_lr: np.ndarray, obs_mask: np.ndarray, t: float) -> np.ndarray:
        """Dự đoán log-tỉ số tại mốc h và h+1: mảng (E, 2)."""
        import torch

        with torch.no_grad():
            return self.model(torch.from_numpy(self.features(obs_lr, obs_mask, t)), self.A).numpy().astype(np.float64)

    def forecast_V(self, pred: np.ndarray, t: float) -> np.ndarray:
        """Ma trận vận tốc dự báo (E,24): mốc h, h+1 theo GCN; các mốc sau tắt dần với hệ số ρ mỗi giờ."""
        h = int((t % 1440.0) // 60.0)
        lr = np.zeros_like(self.V_base)
        lr[:, h] = pred[:, 0]
        for k in range(1, 24):
            lr[:, (h + k) % 24] = pred[:, 1] * (self.rho ** (k - 1))
        return self.V_base * np.exp(lr)

    def observe(self, real: TravelTimeModel, edges: np.ndarray, t: float) -> np.ndarray:
        """log-tỉ số quan sát v_real/v_base = log(τ_base/τ_real) trên các cạnh cho trước."""
        return np.log(self.base.travel_time_vec(edges, t) / real.travel_time_vec(edges, t))

    def model_from_observations(self, real: TravelTimeModel, t: float, extra_edges: np.ndarray | None = None) -> TravelTimeModel:
        """Quan sát cảm biến (+ cạnh cục bộ) tại t → GCN → TravelTimeModel dự báo."""
        mask = self.sensors.copy()
        if extra_edges is not None and len(extra_edges):
            mask[extra_edges] = True
        idx = np.flatnonzero(mask)
        lr = np.zeros(self.graph.n_edges)
        lr[idx] = self.observe(real, idx, t)
        V = self.forecast_V(self.predict(lr, mask, t), t)
        return TravelTimeModel(self.graph.length, V)

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, graph: RoadGraph, V_base: np.ndarray, cfg: Config, path: Path) -> "Forecaster":
        import torch

        f = cls(graph, V_base, cfg)
        f.model.load_state_dict(torch.load(path, map_location="cpu"))
        f.model.eval()
        return f


# ---------------------------------------------------------------------------
def _sample(fc: Forecaster, rng: np.random.Generator, cfg: Config) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Một mẫu huấn luyện: (log-tỉ số quan sát, mặt nạ, t, mục tiêu (E,2))."""
    g = fc.graph
    sigma = float(rng.choice(cfg["scenarios"]["sigma_levels"]))
    V = get_noise_model(g, cfg).real_V(fc.V_base, sigma, int(rng.integers(2**31)))
    t = float(rng.uniform(0, 1440))
    h = int(t // 60.0)
    target = np.column_stack([np.log(V[:, h] / fc.V_base[:, h]), np.log(V[:, (h + 1) % 24] / fc.V_base[:, (h + 1) % 24])])
    mask = fc.sensors.copy()
    node = int(rng.integers(g.n_nodes))
    mask[g.edges_within(node, cfg["algorithms"]["replan"]["observation_radius_m"])] = True
    real = TravelTimeModel(g.length, V)
    idx = np.flatnonzero(mask)
    lr = np.zeros(g.n_edges)
    lr[idx] = fc.observe(real, idx, t)
    return lr, mask, t, target


def evaluate(fc: Forecaster, samples: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]) -> dict[str, float]:
    """RMSE log-tỉ số trên cạnh KHÔNG quan sát: dự báo nền (0), GCN; và trên mọi cạnh."""
    err = {"zero_unobs": [], "gcn_unobs": [], "gcn_all": [], "zero_all": []}
    for lr, mask, t, y in samples:
        p = fc.predict(lr, mask, t)
        un = ~mask
        err["zero_unobs"].append(np.mean(y[un] ** 2))
        err["gcn_unobs"].append(np.mean((p[un] - y[un]) ** 2))
        err["zero_all"].append(np.mean(y ** 2))
        err["gcn_all"].append(np.mean((p - y) ** 2))
    return {k: float(np.sqrt(np.mean(v))) for k, v in err.items()}


def train(cfg: Config) -> dict[str, Any]:
    import torch

    torch.manual_seed(int(cfg["project"]["seed"]))
    graph = RoadGraph.from_config(cfg)
    V_base = load_vbase(cfg)
    fc = Forecaster(graph, V_base, cfg)
    gc = cfg["forecast"]["gcn"]
    rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "gcn_train"))
    vrng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "gcn_val"))
    val = [_sample(fc, vrng, cfg) for _ in range(int(gc["val_fields"]))]
    before = evaluate(fc, val)
    opt = torch.optim.Adam(fc.model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    t0 = time.time()
    hist = []
    for step in range(int(gc["train_steps"])):
        fc.model.train()
        opt.zero_grad()
        loss = 0.0
        for _ in range(int(gc["batch_fields"])):
            lr, mask, t, y = _sample(fc, rng, cfg)
            pred = fc.model(torch.from_numpy(fc.features(lr, mask, t)), fc.A)
            loss = loss + torch.mean((pred - torch.from_numpy(y.astype(np.float32))) ** 2)
        loss = loss / gc["batch_fields"]
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == gc["train_steps"] - 1:
            fc.model.eval()
            ev = evaluate(fc, val)
            hist.append({"step": step, "train_loss": float(loss), **ev})
            log.info("GCN bước %4d | loss %.4f | RMSE chưa quan sát: nền %.4f → GCN %.4f", step, float(loss),
                     ev["zero_unobs"], ev["gcn_unobs"])
    fc.model.eval()
    out_dir = cfg.path("results_dir") / "forecast"
    fc.save(out_dir / "gcn.pt")
    after = evaluate(fc, val)
    summary = {"val_before": before, "val_after": after, "train_s": time.time() - t0,
               "n_sensors": int(fc.sensors.sum()), "sensor_share": float(fc.sensors.mean()), "history": hist}
    save_json(summary, out_dir / "forecast_summary.json")
    log.info("GCN xong (%.0f s): RMSE cạnh chưa quan sát %.4f (nền) → %.4f (GCN); %d cảm biến (%.1f%%)",
             summary["train_s"], after["zero_unobs"], after["gcn_unobs"], summary["n_sensors"], 100 * summary["sensor_share"])
    return summary


def load_forecaster(cfg: Config, graph: RoadGraph, V_base: np.ndarray) -> Forecaster | None:
    p = cfg.path("results_dir") / "forecast" / "gcn.pt"
    return Forecaster.load(graph, V_base, cfg, p) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    train(cfg)


if __name__ == "__main__":
    main()
