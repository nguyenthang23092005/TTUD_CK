"""Giai đoạn 2 — mô phỏng vận tốc 24 khung giờ, nội suy theo phút, FIFO và sự cố.

Mô hình:
    v(e,h) = max(v_free(e) * (1 - c(e,h)), v_min)
    c(e,h) = clip(p(h) * s(group_e) * hotspot(e,h) * direction(e,h) + noise(e,h), 0, c_max)

Thời gian luôn tính bằng PHÚT kể từ 00:00 (có thể > 1440 khi chuyến đi qua nửa đêm; vận tốc nền lặp chu kỳ 24h).

Chạy: python -m src.speed_model [--mode smoke|full]
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from src.config import PROJECT_ROOT, Config, load_config
from src.graph_data import RoadGraph
from src.utils import derive_seed, normalize_text, save_figure, save_json, setup_logging

log = logging.getLogger("speed_model")

KMH_TO_MPM = 1000.0 / 60.0
HOURS = np.arange(24, dtype=np.float64)


# ---------------------------------------------------------------------------
# Thành phần của mô hình ùn tắc
# ---------------------------------------------------------------------------
def wrapped_gaussian(h: np.ndarray | float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian tuần hoàn chu kỳ 24h, đỉnh = 1 tại h = mu."""
    h = np.asarray(h, dtype=np.float64)
    return sum(np.exp(-((h - mu + 24.0 * k) ** 2) / (2.0 * sigma**2)) for k in (-1, 0, 1))


def hourly_profile(cfg: Config, hours: np.ndarray | None = None) -> np.ndarray:
    """p(h) ∈ [0,1]: mức ùn chung theo giờ (tổng Gaussian tuần hoàn)."""
    pc = cfg["speed"]["profile"]
    h = HOURS if hours is None else np.asarray(hours, dtype=np.float64)
    p = np.full_like(h, float(pc["base"]))
    nd = pc.get("night_dip")
    if nd:
        p += nd["amp"] * wrapped_gaussian(h, nd["mu"], nd["sigma"])
    for pk in pc["peaks"]:
        p += pk["amp"] * wrapped_gaussian(h, pk["mu"], pk["sigma"])
    lo, hi = pc["clip"]
    return np.clip(p, lo, hi)


def road_sensitivity(graph: RoadGraph, cfg: Config) -> np.ndarray:
    """s(group) cho từng cạnh."""
    sens = cfg["speed"]["road_sensitivity"]
    table = np.array([float(sens[g]) for g in graph.group_names])
    return table[graph.group_id]


def load_hotspots(path: str | None = None) -> dict[str, Any]:
    """Đọc configs/hotspots.yaml -> {"hotspots": [...], "exclude_name_prefixes": [...]}."""
    p = PROJECT_ROOT / (path or "configs/hotspots.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hour_window_weight(windows: Sequence[Sequence[float]]) -> np.ndarray:
    """w(h) ∈ [0,1] cho 24 giờ: 1 nếu giờ h nằm trong một cửa sổ [start, end) (cho phép vắt qua nửa đêm)."""
    w = np.zeros(24)
    for start, end in windows:
        hs = np.arange(24)
        inside = (hs >= start) & (hs < end) if start <= end else (hs >= start) | (hs < end)
        w = np.maximum(w, inside.astype(float))
    return w


def hotspot_matrix(graph: RoadGraph, spec: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    """Hệ số nhân hotspot(e,h) (E,24) và bảng thống kê số cạnh khớp từng hotspot.

    Khớp theo tên đường/ref (không dấu, chứa chuỗi con) HOẶC theo bán kính quanh toạ độ.
    Nhiều hotspot cùng khớp một cạnh: lấy hệ số lớn nhất.
    """
    df = graph.edges_df
    hotspots = spec["hotspots"]
    excl = tuple(spec.get("exclude_name_prefixes") or ())
    # tên ghép "A / B": bỏ các phần là ngõ/ngách
    names = [" / ".join(part for part in normalize_text(n).split(" / ") if not part.startswith(excl))
             for n in df["name"].fillna("")]
    refs = [normalize_text(r).replace(" ", "").replace(".", "") for r in df["ref"].fillna("")]
    mid_x, mid_y = _edge_mid_xy(graph)
    H = np.ones((graph.n_edges, 24))
    report = []
    for hs in hotspots:
        match = np.zeros(graph.n_edges, dtype=bool)
        keys = [normalize_text(k) for k in hs.get("names", [])]
        if keys:
            match |= np.array([any(k in n for k in keys) for n in names])
        rkeys = [normalize_text(k).replace(" ", "").replace(".", "") for k in hs.get("refs", [])]
        if rkeys:
            match |= np.array([any(k == r or k in r.split("/") for k in rkeys) for r in refs])
        if "center" in hs and hs.get("radius_m"):
            from src.utils import local_xy_m

            cx, cy = local_xy_m(hs["center"][0], hs["center"][1], graph.lat0, graph.lon0)
            dist = np.hypot(mid_x - cx, mid_y - cy)
            if keys and hs.get("name_within_m"):
                match &= dist <= float(hs["name_within_m"])
            match |= dist <= float(hs["radius_m"])
        w = _hour_window_weight(hs["hours"])
        mult = 1.0 + (float(hs["multiplier"]) - 1.0) * w  # (24,)
        H[match] = np.maximum(H[match], mult[None, :])
        report.append({"hotspot": hs["name"], "multiplier": hs["multiplier"], "n_edges": int(match.sum()),
                       "length_km": float(graph.length[match].sum() / 1000), "hours": str(hs["hours"])})
    return H, pd.DataFrame(report)


def _edge_mid_xy(graph: RoadGraph) -> tuple[np.ndarray, np.ndarray]:
    mx = (graph.node_x[graph.edge_from] + graph.node_x[graph.edge_to]) / 2
    my = (graph.node_y[graph.edge_from] + graph.node_y[graph.edge_to]) / 2
    return mx, my


def direction_alignment(graph: RoadGraph, cfg: Config) -> np.ndarray:
    """cos góc giữa hướng cạnh và hướng từ trung điểm cạnh tới tâm tham chiếu (+1: đi vào trung tâm)."""
    from src.utils import local_xy_m

    d = cfg["speed"]["direction"]
    cx, cy = local_xy_m(d["center_lat"], d["center_lon"], graph.lat0, graph.lon0)
    mx, my = _edge_mid_xy(graph)
    ex = graph.node_x[graph.edge_to] - graph.node_x[graph.edge_from]
    ey = graph.node_y[graph.edge_to] - graph.node_y[graph.edge_from]
    tx, ty = cx - mx, cy - my
    norm = np.hypot(ex, ey) * np.hypot(tx, ty)
    return np.where(norm > 0, (ex * tx + ey * ty) / np.maximum(norm, 1e-9), 0.0)


def direction_matrix(graph: RoadGraph, cfg: Config) -> np.ndarray:
    """direction_factor(e,h) = 1 + beta * a_e * (w_sáng(h) - w_chiều(h)), a_e = độ thẳng hàng hướng tâm."""
    d = cfg["speed"]["direction"]
    a = direction_alignment(graph, cfg)
    w_am = wrapped_gaussian(HOURS, d["morning"]["mu"], d["morning"]["sigma"])
    w_pm = wrapped_gaussian(HOURS, d["evening"]["mu"], d["evening"]["sigma"])
    return 1.0 + float(d["beta"]) * a[:, None] * (w_am - w_pm)[None, :]


def edge_adjacency(graph: RoadGraph, self_weight: float = 1.0) -> sp.csr_matrix:
    """Ma trận kề cạnh–cạnh (chung ít nhất một đầu mút), chuẩn hoá theo hàng (trung bình láng giềng)."""
    E, N = graph.n_edges, graph.n_nodes
    rows = np.repeat(np.arange(E), 2)
    cols = np.column_stack([graph.edge_from, graph.edge_to]).ravel()
    B = sp.csr_matrix((np.ones(2 * E), (rows, cols)), shape=(E, N))
    A = (B @ B.T).tocsr()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    A = A + self_weight * sp.identity(E, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / deg) @ A


def spatial_noise(graph: RoadGraph, cfg: Config, seed: int, A: sp.csr_matrix | None = None) -> np.ndarray:
    """Trường nhiễu chuẩn hoá (E,24) có tương quan không gian (qua cạnh kề) và thời gian (qua giờ kề)."""
    nc = cfg["speed"]["spatial_noise"]
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((graph.n_edges, 24))
    A = A if A is not None else edge_adjacency(graph, nc["self_weight"])
    for _ in range(int(nc["smooth_iters"])):
        Z = A @ Z
    k = np.asarray(nc["temporal_kernel"], dtype=np.float64)
    half = len(k) // 2
    for _ in range(int(nc["temporal_iters"])):
        Z = sum(k[i] * np.roll(Z, i - half, axis=1) for i in range(len(k)))
    Z = (Z - Z.mean()) / max(Z.std(), 1e-12)
    return Z


# ---------------------------------------------------------------------------
# FIFO
# ---------------------------------------------------------------------------
def fifo_max_slope(V_kmh: np.ndarray, length_m: np.ndarray) -> np.ndarray:
    """Độ dốc lớn nhất -dτ/dt trên mỗi đoạn giờ (E,24) khi vận tốc nội suy tuyến tính.

    τ(t)=L/v(t), v tuyến tính trong 60 phút -> -τ' = L·Δv/(60·v²), lớn nhất tại đầu mút vận tốc thấp
    (chỉ đoạn vận tốc TĂNG mới làm τ giảm). FIFO ⇔ -τ' ≤ 1.
    """
    v = V_kmh * KMH_TO_MPM
    vn = np.roll(v, -1, axis=1)
    dv = np.maximum(vn - v, 0.0)
    return length_m[:, None] * dv / (60.0 * v**2)


def enforce_fifo(V_kmh: np.ndarray, length_m: np.ndarray, max_slope: float = 0.9,
                 max_iters: int = 200) -> tuple[np.ndarray, int]:
    """Nâng vận tốc đầu mút thấp của các đoạn vi phạm cho tới khi -τ' ≤ max_slope (chỉ tăng vận tốc).

    Returns: (V đã sửa, số phần tử bị chỉnh).
    """
    v = V_kmh.astype(np.float64) * KMH_TO_MPM
    L = length_m[:, None]
    a = 60.0 * max_slope
    n_fixed = 0
    for _ in range(max_iters):
        vn = np.roll(v, -1, axis=1)
        slope = L * np.maximum(vn - v, 0.0) / (60.0 * v**2)
        bad = slope > max_slope * (1 + 1e-9)
        if not bad.any():
            break
        # giải a·x² + L·x − L·v_cao ≥ 0 với x = v_thấp mới
        v_req = (-L + np.sqrt(L**2 + 4 * a * L * vn)) / (2 * a)
        v = np.where(bad, np.maximum(v, v_req * (1 + 1e-6)), v)
        n_fixed += int(bad.sum())
    else:  # pragma: no cover
        log.warning("enforce_fifo chưa hội tụ sau %d vòng", max_iters)
    return v / KMH_TO_MPM, n_fixed


# ---------------------------------------------------------------------------
# Xây dựng V_base
# ---------------------------------------------------------------------------
def build_vbase(graph: RoadGraph, cfg: Config, day_seed: int | None = None) -> dict[str, Any]:
    """Tạo ma trận vận tốc nền (E,24) km/h và các thành phần trung gian."""
    sc = cfg["speed"]
    nc = sc["spatial_noise"]
    seed = int(cfg["project"]["seed"])
    day_seed = int(nc["base_day_seed"]) if day_seed is None else int(day_seed)
    A = edge_adjacency(graph, nc["self_weight"])
    p = hourly_profile(cfg)
    s = road_sensitivity(graph, cfg)
    H, hs_report = hotspot_matrix(graph, load_hotspots(cfg["paths"]["hotspots_file"]))
    Dm = direction_matrix(graph, cfg)
    noise = (float(nc["sigma"]) * spatial_noise(graph, cfg, derive_seed(seed, "spatial"), A)
             + float(nc["day_sigma"]) * spatial_noise(graph, cfg, derive_seed(seed, "day", day_seed), A))
    C = np.clip(p[None, :] * s[:, None] * H * Dm + noise, 0.0, float(sc["c_max"]))
    V = np.maximum(graph.v_free[:, None] * (1.0 - C), float(sc["v_min_kmh"]))
    V, n_fixed = enforce_fifo(V, graph.length, sc["fifo"]["max_slope"], sc["fifo"]["max_iters"])
    return {"V": V, "C": C, "p": p, "s": s, "H": H, "D": Dm, "noise": noise,
            "hotspot_report": hs_report, "fifo_fixed": n_fixed}


# ---------------------------------------------------------------------------
# Sự cố và mô hình thời gian di chuyển
# ---------------------------------------------------------------------------
@dataclass
class Incident:
    """Sự cố trên cạnh `edge`: vận tốc nhân `severity` trong [start, start+duration] (phút tuyệt đối)."""

    edge: int
    start_min: float
    duration_min: float
    severity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def incident_factor(t: np.ndarray, inc: Incident, ramp_in: float, ramp_out: float) -> np.ndarray:
    """Hệ số nhân vận tốc theo thời gian: dốc vào -> mức severity -> giải toả tuyến tính."""
    t = np.asarray(t, dtype=np.float64)
    s = inc.severity
    t0, t1 = inc.start_min, inc.start_min + inc.duration_min
    f = np.ones_like(t)
    up = (t >= t0) & (t < t0 + ramp_in)
    f[up] = 1.0 - (1.0 - s) * (t[up] - t0) / max(ramp_in, 1e-9)
    full = (t >= t0 + ramp_in) & (t <= t1)
    f[full] = s
    down = (t > t1) & (t < t1 + ramp_out)
    f[down] = s + (1.0 - s) * (t[down] - t1) / max(ramp_out, 1e-9)
    return f


def sample_incidents(rng: np.random.Generator, candidates: Sequence[int], n: int, depart_min: float,
                     cfg: Config) -> list[Incident]:
    """Sinh `n` sự cố trên các cạnh ứng viên (không lặp), bắt đầu quanh giờ xuất phát."""
    ic = cfg["speed"]["incidents"]
    off_lo, off_hi = cfg["scenarios"]["incident_start_offset_min"]
    n = min(n, len(candidates))
    if n <= 0:
        return []
    edges = rng.choice(np.asarray(candidates), size=n, replace=False)
    return [Incident(edge=int(e), start_min=float(depart_min + rng.uniform(off_lo, off_hi)),
                     duration_min=float(rng.uniform(*ic["duration_range_min"])),
                     severity=float(rng.uniform(*ic["severity_range"]))) for e in edges]


class TravelTimeModel:
    """Hàm thời gian qua cạnh τ(e,t) = L_e / v(e,t) (phút), v nội suy tuyến tính giữa các khung giờ.

    Sự cố được áp bằng bảng lưới phút và "FIFO hoá" (arrival(t) = min_{t'≥t} arrival(t')),
    đảm bảo tính FIFO tuyệt đối kể cả lúc giải toả (vận tốc tăng đột ngột).
    """

    def __init__(self, length_m: np.ndarray, V_kmh: np.ndarray, incidents: Iterable[Incident] | None = None,
                 ramp_in: float = 5.0, ramp_out: float = 15.0, grid_step: float = 1.0) -> None:
        self.length = np.asarray(length_m, dtype=np.float64)
        self.V_kmh = np.asarray(V_kmh, dtype=np.float64)
        self.V_mpm = self.V_kmh * KMH_TO_MPM
        self.v_max_mpm = float(self.V_mpm.max())
        self.ramp_in, self.ramp_out, self.grid_step = float(ramp_in), float(ramp_out), float(grid_step)
        self.incidents: list[Incident] = list(incidents or [])
        self._len = self.length.tolist()
        self._rows: list[list[float]] | None = None
        self._inc: dict[int, tuple[float, float, float, list[float]]] = {}
        by_edge: dict[int, list[Incident]] = {}
        for inc in self.incidents:
            by_edge.setdefault(inc.edge, []).append(inc)
        for e, incs in by_edge.items():
            self._inc[e] = self._incident_table(e, incs)

    # -- nội bộ ------------------------------------------------------------
    @property
    def rows(self) -> list[list[float]]:
        if self._rows is None:
            self._rows = self.V_mpm.tolist()
        return self._rows

    def base_speed_mpm(self, edges: np.ndarray | int, t: float | np.ndarray) -> np.ndarray:
        """Vận tốc nền (m/phút) nội suy tuyến tính, chu kỳ 24h."""
        tm = np.mod(t, 1440.0)
        h = np.floor(tm / 60.0).astype(np.int64)
        f = (tm - 60.0 * h) / 60.0
        h1 = (h + 1) % 24
        return self.V_mpm[edges, h] * (1.0 - f) + self.V_mpm[edges, h1] * f

    def _incident_table(self, e: int, incs: list[Incident]) -> tuple[float, float, float, list[float]]:
        t_lo = min(i.start_min for i in incs) - self.grid_step
        t_hi = max(i.start_min + i.duration_min for i in incs) + self.ramp_out + 60.0
        grid = np.arange(t_lo, t_hi + self.grid_step, self.grid_step)
        f = np.min([incident_factor(grid, i, self.ramp_in, self.ramp_out) for i in incs], axis=0)
        v = self.base_speed_mpm(np.full(len(grid), e), grid) * f
        arr = grid + self.length[e] / v
        arr = np.minimum.accumulate(arr[::-1])[::-1]  # FIFO hoá (cho phép "chờ" ẩn)
        return float(grid[0]), float(grid[-1]), self.grid_step, arr.tolist()

    # -- API công khai -----------------------------------------------------
    def travel_time(self, e: int, t: float) -> float:
        """τ(e,t) phút — phiên bản scalar thuần Python (dùng trong vòng lặp tìm đường)."""
        tab = self._inc.get(e)
        if tab is not None and tab[0] <= t <= tab[1]:
            t_lo, _, step, arr = tab
            x = (t - t_lo) / step
            i = int(x)
            if i >= len(arr) - 1:
                return arr[-1] - t
            return arr[i] + (arr[i + 1] - arr[i]) * (x - i) - t
        tm = t % 1440.0
        h = int(tm // 60.0)
        fr = (tm - 60.0 * h) / 60.0
        row = self.rows[e]
        v0 = row[h]
        return self._len[e] / (v0 + (row[h + 1 if h < 23 else 0] - v0) * fr)

    def travel_time_vec(self, edges: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """τ cho nhiều cạnh (hoặc nhiều thời điểm) — vector hoá numpy."""
        edges = np.asarray(edges, dtype=np.int64)
        tt = self.length[edges] / self.base_speed_mpm(edges, t)
        if self._inc:
            tarr = np.broadcast_to(np.asarray(t, dtype=np.float64), edges.shape)
            for k in np.flatnonzero(np.isin(edges, list(self._inc))):
                tt[k] = self.travel_time(int(edges[k]), float(tarr[k]))
        return tt

    def speed_kmh(self, edges: np.ndarray, t: float) -> np.ndarray:
        """Vận tốc hiệu dụng (km/h) = L/τ — đã tính sự cố."""
        edges = np.asarray(edges, dtype=np.int64)
        return self.length[edges] / self.travel_time_vec(edges, t) / KMH_TO_MPM

    def path_travel_time(self, edges: Sequence[int], t0: float) -> float:
        """Mô phỏng đi theo dãy cạnh, trả về tổng thời gian (phút)."""
        t = t0
        for e in edges:
            t += self.travel_time(e, t)
        return t - t0


def make_real_V(V_base: np.ndarray, length_m: np.ndarray, sigma: float, seed: int,
                max_slope: float = 0.9) -> np.ndarray:
    """V_real = V_base * eps, eps ~ LogNormal(0, sigma) (E,24) tái lập theo seed, rồi sửa FIFO."""
    if sigma <= 0:
        return V_base.copy()
    eps = np.random.default_rng(seed).lognormal(0.0, sigma, size=V_base.shape)
    V, _ = enforce_fifo(V_base * eps, length_m, max_slope)
    return V


class NoiseModel:
    """Nhiễu log-normal CÓ TƯƠNG QUAN không gian–thời gian cho môi trường thực.

    log ε(e,h) = σ·Z(e,h), Z(·,h) có phân phối biên N(0,1) nên ε vẫn ~ LogNormal(0,σ) như đặc tả, nhưng:
      * không gian: mỗi lớp nhiễu mới (innovation) được làm mượt k lần qua cạnh kề;
      * thời gian: Z(·,h) = ρ·Z(·,h−1) + √(1−ρ²)·W(·,h)  (AR(1) theo giờ);
      * một phần i.i.d. (trọng số `iid_frac`) giữ biến thiên cục bộ.
    Nhờ tương quan, vận tốc quan sát được ở hiện tại/lân cận mang thông tin về tương lai gần và các cạnh chưa quan
    sát — điều kiện cần để dự báo ngắn hạn (GNN) có ý nghĩa.
    """

    def __init__(self, graph: RoadGraph, cfg: Config) -> None:
        nc = cfg["speed"]["real_noise"]
        self.kind = str(nc["model"])
        self.rho = float(nc["rho_hourly"])
        self.k = int(nc["smooth_iters"])
        self.iid_frac = float(nc["iid_frac"])
        self.mean_one = bool(nc.get("mean_one", False))
        self.length = graph.length
        self.max_slope = float(cfg["speed"]["fifo"]["max_slope"])
        self.A = edge_adjacency(graph, cfg["speed"]["spatial_noise"]["self_weight"]) if self.kind == "correlated" else None

    def standard_field(self, seed: int, n_hours: int = 24) -> np.ndarray:
        """Trường Z (E, n_hours) có phân phối biên N(0,1)."""
        rng = np.random.default_rng(seed)
        E = len(self.length)
        if self.kind != "correlated":
            return rng.standard_normal((E, n_hours))
        W = rng.standard_normal((E, n_hours))
        for _ in range(self.k):
            W = self.A @ W
        W /= max(W.std(), 1e-12)
        Z = np.empty_like(W)
        Z[:, 0] = W[:, 0]
        s = math.sqrt(1.0 - self.rho**2)
        for h in range(1, n_hours):
            Z[:, h] = self.rho * Z[:, h - 1] + s * W[:, h]
        Z /= max(Z.std(), 1e-12)
        iid = rng.standard_normal((E, n_hours))
        return math.sqrt(1.0 - self.iid_frac) * Z + math.sqrt(self.iid_frac) * iid

    def log_eps(self, sigma: float, seed: int) -> np.ndarray:
        if sigma <= 0:
            return np.zeros((len(self.length), 24))
        return sigma * self.standard_field(seed) - (0.5 * sigma**2 if self.mean_one else 0.0)

    def real_V(self, V_base: np.ndarray, sigma: float, seed: int) -> np.ndarray:
        """V_real = enforce_fifo(V_base · exp(log ε))."""
        if sigma <= 0:
            return V_base.copy()
        V, _ = enforce_fifo(V_base * np.exp(self.log_eps(sigma, seed)), self.length, self.max_slope)
        return V


_NOISE_CACHE: dict[int, NoiseModel] = {}


def get_noise_model(graph: RoadGraph, cfg: Config) -> NoiseModel:
    """NoiseModel dùng chung (cache theo đồ thị) — tránh dựng lại ma trận kề cho mỗi truy vấn."""
    key = id(graph)
    if key not in _NOISE_CACHE:
        _NOISE_CACHE[key] = NoiseModel(graph, cfg)
    return _NOISE_CACHE[key]


def real_V(graph: RoadGraph, V_base: np.ndarray, sigma: float, seed: int, cfg: Config) -> np.ndarray:
    """Sinh V_real theo mô hình nhiễu trong cấu hình (`speed.real_noise.model`: iid | correlated)."""
    return get_noise_model(graph, cfg).real_V(V_base, sigma, seed)


def check_fifo_numeric(model: TravelTimeModel, edges: Sequence[int], t_grid: np.ndarray) -> float:
    """Trả về độ giảm lớn nhất của arrival(t)=t+τ(e,t) trên lưới (<=0 nghĩa là FIFO thoả)."""
    worst = -np.inf
    for e in edges:
        arr = np.array([t + model.travel_time(int(e), float(t)) for t in t_grid])
        worst = max(worst, float(np.max(arr[:-1] - arr[1:])))
    return worst


# ---------------------------------------------------------------------------
# Lưu / nạp
# ---------------------------------------------------------------------------
def load_vbase(cfg: Config) -> np.ndarray:
    return np.load(cfg.path("speeds_dir") / "V_base.npy")


def base_model(graph: RoadGraph, V_base: np.ndarray, cfg: Config,
               incidents: Iterable[Incident] | None = None) -> TravelTimeModel:
    ic = cfg["speed"]["incidents"]
    return TravelTimeModel(graph.length, V_base, incidents, ic["ramp_in_min"], ic["ramp_out_min"],
                           ic["fifo_grid_step_min"])


# ---------------------------------------------------------------------------
# Biểu đồ
# ---------------------------------------------------------------------------
def plot_speed_maps(graph: RoadGraph, V: np.ndarray, hours: Sequence[int], cfg: Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from src.graph_builder import GROUP_WIDTH

    segs = [c[:, ::-1] for c in graph.edge_coords]
    widths = np.array([GROUP_WIDTH[graph.group_names[g]] for g in graph.group_id]) * 0.9 + 0.2
    order = np.argsort(-graph.group_id)  # vẽ đường nhỏ trước
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    cmap = plt.get_cmap("RdYlGn")
    norm = matplotlib.colors.Normalize(0.15, 1.0)
    for ax, h in zip(axes.ravel(), hours):
        ratio = V[:, h] / graph.v_free
        lc = LineCollection([segs[i] for i in order], colors=cmap(norm(ratio[order])), linewidths=widths[order])
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect(1 / math.cos(math.radians(graph.lat0)))
        ax.set_title(f"{h:02d}:00 — v/v_free TB = {ratio.mean():.2f}, v TB = {V[:, h].mean():.1f} km/h", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Tỉ lệ vận tốc v(e,h) / v_free(e)")
    save_figure(fig, cfg.path("figures_dir"), "speed_maps", cfg["evaluation"]["figure_dpi"])
    plt.close(fig)


def plot_speed_curves(graph: RoadGraph, V: np.ndarray, p: np.ndarray, cfg: Config) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for gi, g in enumerate(graph.group_names):
        m = graph.group_id == gi
        if m.any():
            w = graph.length[m]
            for h in range(24):
                rows.append({"group": g, "hour": h, "mean_speed_kmh": float(np.average(V[m, h], weights=w))})
    df = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for g, sub in df.groupby("group", sort=False):
        ax1.plot(sub.hour, sub.mean_speed_kmh, marker="o", ms=3, label=g)
    ax1.set_xlabel("Giờ trong ngày")
    ax1.set_ylabel("Vận tốc TB (km/h, trọng số chiều dài)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("Vận tốc nền theo giờ và loại đường")
    ax2.plot(HOURS, p, color="k", marker="o", ms=3)
    ax2.set_xlabel("Giờ trong ngày")
    ax2.set_ylabel("p(h)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(alpha=0.3)
    ax2.set_title("Profile ùn tắc chung p(h) (chu kỳ 24h)")
    save_figure(fig, cfg.path("figures_dir"), "speed_curves", cfg["evaluation"]["figure_dpi"])
    plt.close(fig)
    return df


def run(cfg: Config) -> dict[str, Any]:
    """Chạy giai đoạn 2: tạo V_base, kiểm tra FIFO, vẽ hình."""
    graph = RoadGraph.from_config(cfg)
    out = build_vbase(graph, cfg)
    V = out["V"]
    sdir = cfg.path("speeds_dir")
    np.save(sdir / "V_base.npy", V.astype(np.float64))
    np.save(sdir / "C_base.npy", out["C"].astype(np.float32))
    pd.DataFrame({"hour": HOURS.astype(int), "p": out["p"]}).to_csv(sdir / "profile.csv", index=False)
    out["hotspot_report"].to_csv(cfg.path("results_dir") / "hotspot_matches.csv", index=False)

    slope = fifo_max_slope(V, graph.length)
    model = base_model(graph, V, cfg)
    rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "fifo_check"))
    sample = rng.choice(graph.n_edges, size=min(300, graph.n_edges), replace=False)
    numeric = check_fifo_numeric(model, sample, np.arange(0.0, 1440.0, 1.0))
    curves = plot_speed_curves(graph, V, out["p"], cfg)
    curves.to_csv(cfg.path("results_dir") / "speed_by_hour_group.csv", index=False)
    plot_speed_maps(graph, V, cfg["speed"]["plot_hours"], cfg)

    w = graph.length
    stats = {
        "mean_speed_by_hour_kmh": [float(np.average(V[:, h], weights=w)) for h in range(24)],
        "mean_congestion_by_hour": [float(np.average(out["C"][:, h], weights=w)) for h in range(24)],
        "global_v_max_kmh": float(V.max()), "global_v_min_kmh": float(V.min()),
        "fifo_cells_fixed": int(out["fifo_fixed"]),
        "fifo_max_slope_analytic": float(slope.max()),
        "fifo_numeric_worst_decrease_min": numeric,
        "hotspots": out["hotspot_report"].to_dict(orient="records"),
    }
    save_json(stats, cfg.path("results_dir") / "speed_stats.json")
    log.info("V_base: v TB 03h=%.1f, 08h=%.1f, 14h=%.1f, 18h=%.1f km/h; FIFO sửa %d ô; slope max %.3f; "
             "kiểm tra số: giảm lớn nhất %.2e phút", *(stats["mean_speed_by_hour_kmh"][h] for h in (3, 8, 14, 18)),
             stats["fifo_cells_fixed"], stats["fifo_max_slope_analytic"], numeric)
    for r in stats["hotspots"]:
        log.info("  hotspot %-28s x%.2f: %4d cạnh (%.1f km)", r["hotspot"], r["multiplier"], r["n_edges"], r["length_km"])
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg)


if __name__ == "__main__":
    main()
