"""Tổng hợp kết quả: biểu đồ (PNG 300 dpi + PDF), bảng CSV/LaTeX và macro số liệu cho báo cáo.

Mọi con số đọc từ results/ (do các giai đoạn trước sinh ra), không gõ tay.

Chạy: python -m src.evaluation.plots [--mode smoke|full]
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.evaluation.metrics import (CLASSIC_METHODS, MACRO_KEY, METHOD_LABELS, RL_METHODS, TRIP_METHODS,  # noqa: E402
                                    add_regret, breakdown, paired_tests, summarize, to_latex_table, write_macros)
from src.utils import load_json, save_figure, setup_logging  # noqa: E402

log = logging.getLogger("plots")
_cmap = plt.get_cmap("tab20")
PALETTE = {m: _cmap(i % 20) for i, m in enumerate(METHOD_LABELS)}
PALETTE.update({"oracle": "#000000", "dijkstra_length": "#7f7f7f", "plan_once": "#1f77b4", "replan": "#2ca02c",
                "rl": "#d62728", "rl_hybrid": "#9467bd", "rl_tree_bc": "#ff7f0e"})
BREAKDOWN_METHODS = ["dijkstra_length", "plan_once", "plan_once_gnn", "replan", "replan_gnn", "rl_hybrid", "rl_tree_bc"]
SUMMARY_COLS = {"label": "Phương pháp", "success_pct": "Tới đích (\\%)", "tt_mean": "TB (phút)", "tt_median": "Trung vị",
                "tt_p90": "P90", "regret_mean": "Regret TB (\\%)", "regret_median": "Regret TV (\\%)",
                "regret_p90": "Regret P90 (\\%)", "length_km": "Dài (km)"}


class Ctx:
    """Ngữ cảnh: thư mục, macro."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.res = cfg.path("results_dir")
        self.fig = cfg.path("figures_dir")
        self.tab = cfg.path("tables_dir")
        self.dpi = int(cfg["evaluation"]["figure_dpi"])
        self.macros: dict[str, Any] = {}

    def save(self, fig: plt.Figure, name: str) -> None:
        save_figure(fig, self.fig, name, self.dpi)
        plt.close(fig)

    def table(self, df: pd.DataFrame, name: str, columns: dict[str, str], caption: str, **kw: Any) -> None:
        df.to_csv(self.tab / f"{name}.csv", index=False)
        to_latex_table(df, columns, self.tab / f"{name}.tex", caption, f"tab:{name}", **kw)


def _present(df: pd.DataFrame, methods: list[str]) -> list[str]:
    have = set(df.method)
    return [m for m in methods if m in have]


def _rotate(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", rotation=35)
    for t in ax.get_xticklabels():
        t.set_ha("right")


def _f(v: float, d: int = 2) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# ---------------------------------------------------------------------------
# Dữ liệu & mô phỏng
# ---------------------------------------------------------------------------
def graph_tables(c: Ctx) -> None:
    gs = load_json(c.res / "graph_stats.json")
    rows = [{"group": g, "n_edges": v["n_edges"], "length_km": v["length_km"],
             "share_pct": 100 * v["length_km"] / gs["total_length_km"]} for g, v in gs["by_group"].items()]
    rows.append({"group": "Tổng", "n_edges": gs["n_edges"], "length_km": gs["total_length_km"], "share_pct": 100.0})
    c.table(pd.DataFrame(rows), "tab_graph_groups", {"group": "Nhóm đường", "n_edges": "Số cạnh",
            "length_km": "Chiều dài (km)", "share_pct": "Tỉ lệ (\\%)"}, "Phân bố loại đường của đồ thị Hà Đông (sau làm sạch).",
            digits={"length_km": 1, "share_pct": 1})
    c.macros.update(NNodes=gs["n_nodes"], NEdges=gs["n_edges"], TotalKm=f"{gs['total_length_km']:.1f}",
                    RawNodes=gs["raw_nodes"], RawEdges=gs["raw_edges"], SelfLoops=gs["self_loops_removed"],
                    ParallelEdges=gs["parallel_edges_collapsed"], DroppedNodes=gs["nodes_dropped_not_in_scc"],
                    MaxOutDeg=gs["max_out_degree"], MeanOutDeg=f"{gs['mean_out_degree']:.2f}",
                    MaxspeedCoverage=f"{100 * gs['maxspeed_tag_coverage']:.1f}", MeanEdgeLen=f"{gs['mean_edge_length_m']:.0f}")
    from src.graph_data import RoadGraph

    g = RoadGraph.from_config(c.cfg)
    nb = [{v for _, v in g.adj[u]} | {v for _, v in g.radj[u]} for u in range(g.n_nodes)]
    dead = sum(1 for x in nb if len(x) == 1)
    c.macros.update(NDeadEnds=dead, DeadEndPct=f"{100 * dead / g.n_nodes:.1f}")
    ss = load_json(c.res / "speed_stats.json")
    v = ss["mean_speed_by_hour_kmh"]
    c.macros.update(VThree=f"{v[3]:.1f}", VEight=f"{v[8]:.1f}", VFourteen=f"{v[14]:.1f}", VEighteen=f"{v[18]:.1f}",
                    FifoSlope=f"{ss['fifo_max_slope_analytic']:.3f}", FifoFixed=ss["fifo_cells_fixed"],
                    VMaxGlobal=f"{ss['global_v_max_kmh']:.1f}")
    hs = pd.read_csv(c.res / "hotspot_matches.csv")
    c.table(hs, "tab_hotspots", {"hotspot": "Điểm/trục nóng", "multiplier": "Hệ số", "hours": "Khung giờ [bắt đầu, kết thúc)",
            "n_edges": "Số cạnh khớp", "length_km": "Chiều dài (km)"},
            "Các điểm nóng (giả định cần hiệu chỉnh) và số cạnh được khớp.", digits={"multiplier": 2, "length_km": 1}, align="lrlrr")
    sc = load_json(c.res / "scenarios_summary.json")
    c.macros.update(NTrain=sc["n"]["train"], NVal=sc["n"]["val"], NTest=sc["n"]["test"],
                    TestMeanDistKm=f"{sc['mean_od_dist_m']['test'] / 1000:.2f}", TestMeanInc=f"{sc['mean_incidents']['test']:.2f}")
    _noise_stats(c, g)


def _noise_stats(c: Ctx, g: Any) -> None:
    """Thống kê mô hình nhiễu tương quan (đo trực tiếp trên một trường mẫu)."""
    from src.speed_model import get_noise_model

    nm = get_noise_model(g, c.cfg)
    Z = nm.standard_field(12345)
    e = np.arange(0, g.n_edges, 3)
    nxt = np.array([g.out_of(int(g.edge_to[i]))[0] for i in e])
    c.macros.update(NoiseLagCorr=f"{np.corrcoef(Z[:, :-1].ravel(), Z[:, 1:].ravel())[0, 1]:.2f}",
                    NoiseAdjCorr=f"{np.corrcoef(Z[e, 8], Z[nxt, 8])[0, 1]:.2f}",
                    NoiseRho=c.cfg["speed"]["real_noise"]["rho_hourly"], NoiseIid=c.cfg["speed"]["real_noise"]["iid_frac"])


def config_table(c: Ctx) -> None:
    cfg = c.cfg
    rl = cfg["rl"]
    rows = [
        ("graph.buffer\\_m", cfg["graph"]["buffer_m"], "vùng đệm quanh ranh giới quận (m)"),
        ("graph.free\\_flow\\_factor", cfg["graph"]["free_flow_factor"], "v\\_free = maxspeed × hệ số"),
        ("speed.c\\_max / v\\_min", f"{cfg['speed']['c_max']} / {cfg['speed']['v_min_kmh']}", "chặn trên ùn / vận tốc tối thiểu (km/h)"),
        ("speed.profile.peaks", "; ".join(f"{p['name']}: μ={p['mu']}, σ={p['sigma']}, a={p['amp']}" for p in cfg["speed"]["profile"]["peaks"]), "Gaussian tuần hoàn"),
        ("speed.road\\_sensitivity", ", ".join(f"{k}:{v}" for k, v in cfg["speed"]["road_sensitivity"].items()), "s(nhóm đường)"),
        ("speed.direction.beta", cfg["speed"]["direction"]["beta"], "biên độ hệ số hướng"),
        ("speed.real\\_noise", f"{cfg['speed']['real_noise']['model']}, ρ={cfg['speed']['real_noise']['rho_hourly']}, k={cfg['speed']['real_noise']['smooth_iters']}, iid={cfg['speed']['real_noise']['iid_frac']}", "nhiễu môi trường thực"),
        ("speed.fifo.max\\_slope", cfg["speed"]["fifo"]["max_slope"], "giới hạn $-d\\tau/dt$"),
        ("speed.incidents", f"hệ số {cfg['speed']['incidents']['severity_range']}, {cfg['speed']['incidents']['duration_range_min']} phút", "sự cố"),
        ("scenarios.n\\_queries", cfg["scenarios"]["n_queries"], "số truy vấn (full)"),
        ("scenarios.sigma\\_levels", cfg["scenarios"]["sigma_levels"], "mức nhiễu"),
        ("scenarios.*on\\_route\\_prob", f"chính {cfg['scenarios']['incident_on_route_prob']}, stress {cfg['scenarios']['stress_on_route_prob']}", "xác suất đặt sự cố trên tuyến"),
        ("algorithms.replan", f"r={cfg['algorithms']['replan']['observation_radius_m']} m, lười θ={cfg['algorithms']['replan']['lazy_threshold']}", "quan sát / replan lười"),
        ("algorithms.dstar.refresh\\_min", cfg["algorithms"]["dstar"]["refresh_min"], "chu kỳ dựng lại D* Lite (phút)"),
        ("algorithms.alt.n\\_landmarks", cfg["algorithms"]["alt"]["n_landmarks"], "số điểm mốc ALT"),
        ("algorithms.risk", f"M={cfg['algorithms']['risk']['n_samples']}, k={cfg['algorithms']['risk']['k_candidates']}, q={cfg['algorithms']['risk']['quantile']}", "định tuyến P90"),
        ("forecast", f"cảm biến {cfg['forecast']['sensor_fraction']}, GCN {cfg['forecast']['gcn']['layers']}×{cfg['forecast']['gcn']['hidden']}, {cfg['forecast']['gcn']['train_steps']} bước", "dự báo ngắn hạn"),
        ("rl.ppo", f"lr={rl['ppo']['learning_rate']}, n_steps={rl['ppo']['n_steps']}×{rl['n_envs']}, batch={rl['ppo']['batch_size']}, γ={rl['ppo']['gamma']}, ent={rl['ppo']['ent_coef']}", "MaskablePPO"),
        ("rl.variants", "; ".join(f"{v['name']}={v.get('timesteps', rl['total_timesteps'])}" for v in rl["variants"]), "bước huấn luyện / seed"),
        ("rl.bc", f"{rl['bc']['n_episodes']} tập, {rl['bc']['epochs']} epoch, khám phá {rl['bc']['explore_prob']}", "behavior cloning"),
        ("env.reward", ", ".join(f"{k}={v}" for k, v in cfg["env"]["reward"].items()), "hệ số thưởng"),
        ("departure", f"bước {cfg['departure']['wait_step_min']} phút ×{cfg['departure']['max_waits']}, w={cfg['departure']['wait_cost_weights']}", "chọn giờ xuất phát"),
        ("multiagent", f"{cfg['multiagent']['vehicles_per_agent']} xe/tác tử, BPR α={cfg['multiagent']['bpr_alpha']}, β={cfg['multiagent']['bpr_beta']}", "mô phỏng nhiều xe"),
    ]
    df = pd.DataFrame(rows, columns=["key", "value", "desc"])
    df["value"] = df["value"].astype(str).str.replace("_", "\\_")
    df.to_csv(c.tab / "tab_config.csv", index=False)
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\caption{Tham số cấu hình chính (configs/config.yaml).}",
             r"\label{tab:tab_config}", r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}p{5.0cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{3.0cm}}", r"\toprule", r"Khoá & Giá trị & Ý nghĩa \\", r"\midrule"]
    lines += [f"\\texttt{{{r.key}}} & {r.value} & {r.desc} \\\\" for r in df.itertuples()]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (c.tab / "tab_config.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    c.macros.update(Seed=cfg["project"]["seed"], ReplanRadius=cfg["algorithms"]["replan"]["observation_radius_m"],
                    LazyTheta=cfg["algorithms"]["replan"]["lazy_threshold"], DstarRefresh=cfg["algorithms"]["dstar"]["refresh_min"],
                    NLandmarks=cfg["algorithms"]["alt"]["n_landmarks"], RiskM=cfg["algorithms"]["risk"]["n_samples"],
                    RiskK=cfg["algorithms"]["risk"]["k_candidates"], SensorFrac=f"{100 * cfg['forecast']['sensor_fraction']:.0f}",
                    GcnRecompute=cfg["forecast"]["recompute_min"], WaitStep=cfg["departure"]["wait_step_min"],
                    MaxWaits=cfg["departure"]["max_waits"], TreeDepth=cfg["env"]["tree_depth"],
                    StressProb=cfg["scenarios"]["stress_on_route_prob"])


# ---------------------------------------------------------------------------
# Kết quả chính
# ---------------------------------------------------------------------------
def _method_macros(c: Ctx, summ: pd.DataFrame, prefix: str = "") -> None:
    s = summ.set_index("method")
    for m in s.index:
        k = prefix + MACRO_KEY.get(m, m)
        c.macros[f"{k}TT"] = _f(s.loc[m, "tt_mean"])
        c.macros[f"{k}TTPn"] = _f(s.loc[m, "tt_p90"])
        c.macros[f"{k}Regret"] = _f(s.loc[m, "regret_mean"])
        c.macros[f"{k}RegretMed"] = _f(s.loc[m, "regret_median"])
        c.macros[f"{k}RegretPn"] = _f(s.loc[m, "regret_p90"])
        c.macros[f"{k}Success"] = _f(s.loc[m, "success_pct"], 1)


def main_results(c: Ctx) -> pd.DataFrame:
    ec = c.cfg["evaluation"]
    df = add_regret(pd.read_csv(c.res / "benchmark_per_query.csv"))
    df.to_csv(c.res / "benchmark_with_regret.csv", index=False)
    nq = df.query_id.nunique()
    classic = summarize(df, _present(df, CLASSIC_METHODS))
    c.table(classic, "tab_main_classic", SUMMARY_COLS, f"Kết quả các phương pháp tìm kiếm và dự báo trên tập kiểm tra chính ({nq} truy vấn, "
            "sự cố đặt ngẫu nhiên) trong môi trường thực; regret tính so với oracle.", digits=2, align="lrrrrrrrr")
    rl = summarize(df, _present(df, RL_METHODS + ["replan", "oracle"]))
    c.table(rl, "tab_main_rl", SUMMARY_COLS, f"Kết quả các biến thể học tăng cường (seed tốt nhất theo tập kiểm định) và baseline tham lam trên cùng tập "
            "kiểm tra; thời gian và regret chỉ tính trên các chuyến tới đích.", digits=2, align="lrrrrrrrr")
    _method_macros(c, summarize(df, _present(df, TRIP_METHODS)))
    noinc = df[df.n_incidents == 0]
    c.table(summarize(noinc, _present(noinc, ["dijkstra_length", "freeflow", "plan_once", "plan_once_gnn", "replan",
                                               "replan_gnn", "rl_hybrid", "oracle"])),
            "tab_results_noinc", SUMMARY_COLS, f"Kết quả trên tập con kiểm tra không có sự cố ({noinc.query_id.nunique()} truy vấn).",
            digits=2, align="lrrrrrrrr")
    c.macros["NTestNoInc"] = noinc.query_id.nunique()
    tests = paired_tests(df, ec["pairs"], int(ec["bootstrap_samples"]), float(ec["ci_level"]), int(c.cfg["project"]["seed"]))
    tests["pair"] = tests.A.map(METHOD_LABELS) + " − " + tests.B.map(METHOD_LABELS)
    tests["p_str"] = tests.p_value.map(lambda p: "$<10^{-4}$" if p < 1e-4 else f"{p:.4f}".replace(".", "{,}"))
    c.table(tests, "tab_stat_tests", {"pair": "Cặp (A − B)", "n_pairs": "n", "mean_diff_min": "Hiệu TB (phút)",
            "ci_lo": "CI 95\\% thấp", "ci_hi": "CI 95\\% cao", "mean_diff_pct": "Hiệu TB (\\%)",
            "A_better_pct": "A tốt hơn (\\%)", "p_str": "p (Wilcoxon)"},
            "So sánh ghép cặp trên các truy vấn cả hai phương pháp cùng tới đích: khoảng tin cậy bootstrap 95\\% của hiệu "
            "trung bình thời gian và kiểm định Wilcoxon signed-rank.",
            digits=3, align="lrrrrrrr")
    for r in tests.itertuples():
        key = "".join(w.capitalize() for w in f"{r.A} {r.B}".replace("_", " ").split())
        key = "".join(ch for ch in key if ch.isalpha())
        c.macros[f"Diff{key}"] = f"{r.mean_diff_min:.2f}"
        vn = lambda x: f"{x:.2f}".replace(".", "{,}")  # noqa: E731
        c.macros[f"Ci{key}"] = f"$[{vn(r.ci_lo)}$; ${vn(r.ci_hi)}]$"  # tách 2 mảnh để ngắt dòng được
        c.macros[f"P{key}"] = r.p_str if r.p_value < 1e-4 else f"$={r.p_str}$"
        c.macros[f"Better{key}"] = f"{r.A_better_pct:.1f}"
    # số lần tìm kiếm của replan / lười / D*
    for m in ("replan", "replan_lazy", "replan_dstar", "replan_gnn"):
        if m in set(df.method):
            k = MACRO_KEY[m]
            c.macros[f"{k}Replans"] = f"{df[df.method == m].n_replans.mean():.1f}"
            c.macros[f"{k}Expanded"] = f"{df[df.method == m].expanded.mean():.0f}"
    if "same_as_plan_once" in df:
        c.macros["RiskSamePct"] = f"{100 * df[df.method == 'risk_p90'].same_as_plan_once.astype(float).mean():.1f}"
    if "mean_beta" in df and (df.method == "rl_hybrid").any():
        c.macros["HybMeanBeta"] = f"{df[df.method == 'rl_hybrid'].mean_beta.mean():.2f}"
    return df


def stress_results(c: Ctx) -> None:
    p = c.res / "benchmark_stress.csv"
    if not p.exists():
        return
    ds = add_regret(pd.read_csv(p))
    summ = summarize(ds, _present(ds, ["dijkstra_length", "freeflow", "plan_once", "plan_once_gnn", "risk_p90", "replan",
                                       "replan_lazy", "replan_dstar", "replan_gnn", "rl_hybrid", "rl_tree_bc", "oracle"]))
    c.table(summ, "tab_stress", SUMMARY_COLS, f"Kết quả trên tập stress (sự cố được ưu tiên đặt trên tuyến dự kiến với xác suất "
            f"{c.cfg['scenarios']['stress_on_route_prob']}): cùng {ds.query_id.nunique()} truy vấn, cùng thời điểm/mức độ sự cố.",
            digits=2, align="lrrrrrrrr")
    _method_macros(c, summ, prefix="S")


def breakdown_tables(c: Ctx, df: pd.DataFrame) -> None:
    methods = _present(df, BREAKDOWN_METHODS)
    df = df.copy()
    df["peak_label"] = np.where(df.peak, "cao điểm", "thấp điểm")
    labels = c.cfg["scenarios"]["distance_bin_labels"]
    df["dist_label"] = df.distance_bin.map(lambda b: labels[int(b)])
    for by, name, cap in (("peak_label", "tab_by_peak", "giờ xuất phát (cao điểm: " + ", ".join(f"{h}h" for h in c.cfg["evaluation"]["peak_hours"]) + ")"),
                          ("sigma", "tab_by_sigma", "mức nhiễu $\\sigma$"), ("n_incidents", "tab_by_incidents", "số sự cố"),
                          ("dist_label", "tab_by_distance", "khoảng cách O–D")):
        b = breakdown(df, by, methods)
        piv = b.pivot(index=by, columns="method", values="regret_mean")[methods].reset_index()
        piv = b.groupby(by)["n"].max().reset_index().merge(piv, on=by)
        piv[by] = piv[by].astype(str)
        b.to_csv(c.tab / f"{name}_long.csv", index=False)
        c.table(piv, name, {by: "Nhóm", "n": "n"} | {m: METHOD_LABELS[m] for m in methods},
                f"Regret trung bình (\\%) so với oracle theo {cap} (chuyến tới đích).", digits=2, align="lr" + "r" * len(methods))
    b = breakdown(df, "n_incidents", ["plan_once", "replan", "replan_gnn"]).pivot(index="n_incidents", columns="method", values="regret_mean")
    mx = int(b.index.max())
    c.macros.update(MaxInc=mx, PoRegretMaxInc=_f(b.loc[mx, "plan_once"]), RpRegretMaxInc=_f(b.loc[mx, "replan"]))
    if 0 in b.index:
        c.macros.update(PoRegretZeroInc=_f(b.loc[0, "plan_once"]), RpRegretZeroInc=_f(b.loc[0, "replan"]))
    bp = breakdown(df, "peak_label", ["plan_once", "dijkstra_length", "replan"]).pivot(index="peak_label", columns="method", values="regret_mean")
    for lab, key in (("cao điểm", "Peak"), ("thấp điểm", "Off")):
        if lab in bp.index:
            c.macros.update({f"DijRegret{key}": _f(bp.loc[lab, "dijkstra_length"]), f"PoRegret{key}": _f(bp.loc[lab, "plan_once"]),
                             f"RpRegret{key}": _f(bp.loc[lab, "replan"])})


def route_overlap(c: Ctx, df: pd.DataFrame) -> None:
    e = df.pivot_table(index="query_id", columns="method", values="edges", aggfunc="first")
    pairs = [("rl_ctg", "freeflow"), ("rl_tree_bc", "oracle"), ("rl_tree_bc", "plan_once"), ("rl_hybrid", "replan"),
             ("plan_once", "oracle"), ("replan", "oracle"), ("replan_lazy", "replan"), ("replan_dstar", "replan"),
             ("replan_gnn", "replan"), ("risk_p90", "plan_once"), ("freeflow", "dijkstra_length")]
    rows = []
    for a, b in pairs:
        if a not in e or b not in e:
            continue
        both = e[[a, b]].dropna()
        both = both[(both[a].str.len() > 0) & (both[b].str.len() > 0)]
        if both.empty:
            continue
        jac = [len(set(x.split()) & set(y.split())) / len(set(x.split()) | set(y.split())) for x, y in zip(both[a], both[b])]
        same = 100 * float((both[a] == both[b]).mean())
        rows.append({"pair": f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}", "n": len(both), "identical_pct": same,
                     "jaccard_pct": 100 * float(np.mean(jac))})
        key = MACRO_KEY[a] + MACRO_KEY[b]
        c.macros[f"RouteSame{key}"] = f"{same:.1f}"
        c.macros[f"RouteJac{key}"] = f"{100 * np.mean(jac):.1f}"
    c.table(pd.DataFrame(rows), "tab_route_overlap", {"pair": "Cặp phương pháp", "n": "n", "identical_pct": "Tuyến giống hệt (\\%)",
            "jaccard_pct": "Jaccard cạnh TB (\\%)"}, "Độ trùng tuyến giữa các cặp phương pháp trên tập kiểm tra chính.",
            digits={"n": 0, "identical_pct": 1, "jaccard_pct": 1}, align="lrrr")


# ---------------------------------------------------------------------------
# Hình
# ---------------------------------------------------------------------------
def fig_boxplots(c: Ctx, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [11, 9]})
    for ax, methods, title in ((axes[0], CLASSIC_METHODS[:-1], "Tìm kiếm / dự báo"), (axes[1], RL_METHODS, "RL & baseline cục bộ")):
        ms = _present(df, methods)
        ok = df[df.method.isin(ms) & df.reached].copy()
        ok["label"] = ok.method.map(METHOD_LABELS)
        order = [METHOD_LABELS[m] for m in ms]
        sns.boxplot(data=ok, x="label", y="regret_pct", hue="label", order=order,
                    palette={METHOD_LABELS[m]: PALETTE[m] for m in ms}, ax=ax, showfliers=False, legend=False)
        ax.set_yscale("symlog", linthresh=5)
        ax.set_ylabel("Regret so với oracle (%, symlog)")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        _rotate(ax)
    fig.suptitle("Phân bố regret trên tập kiểm tra chính (chuyến tới đích; bỏ ngoại lai)", fontsize=11)
    fig.tight_layout()
    c.save(fig, "boxplot_methods")


def fig_regret_by_hour(c: Ctx, df: pd.DataFrame) -> None:
    methods = _present(df, ["dijkstra_length", "freeflow", "plan_once", "plan_once_gnn", "replan", "replan_gnn", "rl_hybrid"])
    g = df[df.method.isin(methods) & df.reached].groupby(["hour", "method"]).regret_pct.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in methods:
        s = g[g.method == m]
        ax.plot(s.hour, s.regret_pct, marker="o", ms=3, label=METHOD_LABELS[m], color=PALETTE[m])
    n_h = df[df.method == "oracle"].groupby("hour").size()
    ax.set(xlabel="Giờ xuất phát", ylabel="Regret TB (%)", xticks=range(0, 24, 2),
           title=f"Regret TB theo giờ xuất phát (tập kiểm tra; {n_h.min()}–{n_h.max()} truy vấn/giờ)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    c.save(fig, "regret_by_hour")


def fig_breakdowns(c: Ctx, df: pd.DataFrame) -> None:
    methods = _present(df, BREAKDOWN_METHODS)
    ok = df[df.method.isin(methods) & df.reached].copy()
    labels = c.cfg["scenarios"]["distance_bin_labels"]
    ok["dist_label"] = ok.distance_bin.map(lambda b: labels[int(b)])
    ok["label"] = ok.method.map(METHOD_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, by, title in zip(axes, ["n_incidents", "sigma", "dist_label"], ["Số sự cố", "Mức nhiễu σ", "Khoảng cách O–D"]):
        order = sorted(ok[by].unique()) if by != "dist_label" else labels
        sns.barplot(data=ok, x=by, y="regret_pct", hue="label", order=order, hue_order=[METHOD_LABELS[m] for m in methods],
                    palette={METHOD_LABELS[m]: PALETTE[m] for m in methods}, ax=ax, errorbar=("ci", 95), n_boot=500, seed=0)
        ax.set_yscale("symlog", linthresh=2)
        ax.set(xlabel=title, ylabel="Regret TB (%) ± 95% CI (symlog)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    c.save(fig, "regret_breakdowns")


def runtime_tables(c: Ctx, df: pd.DataFrame) -> None:
    p = c.res / "runtime_single.csv"
    if not p.exists():
        return
    rt = pd.read_csv(p)
    order = ["dijkstra_length", "astar_length", "freeflow", "td_dijkstra", "plan_once", "td_astar_lb", "td_astar_alt",
             "td_astar_numba", "oracle", "plan_once_sensor", "plan_once_gnn", "risk_p90", "replan", "replan_lazy",
             "replan_dstar", "replan_gnn", "greedy_local"] + [m for m in RL_METHODS if m != "greedy_local"]
    rt = rt.set_index("method").reindex([m for m in order if m in set(rt.method)]).reset_index()
    rt["label"] = rt.method.map(METHOD_LABELS)
    rt.loc[rt.method == "plan_once", "label"] = "TD-A* (chim bay) = plan-once"
    if "prep_ms" not in rt:
        rt["prep_ms"] = np.nan
    c.table(rt, "tab_runtime_single", {"label": "Phương pháp", "expanded": "Nút mở rộng / bước TB", "runtime_ms": "Thời gian TB (ms)",
            "runtime_p90_ms": "P90 (ms)", "prep_ms": "Tiền xử lý / truy vấn (ms)"},
            f"Chi phí tính toán đơn luồng (một tiến trình, tuần tự, {int(rt.n_queries.iloc[0])} truy vấn kiểm tra, sau khi làm nóng). "
            "Với RL, cột nút mở rộng là số bước chính sách.", digits={"expanded": 0, "runtime_ms": 2, "runtime_p90_ms": 2, "prep_ms": 2},
            align="lrrrr")
    e = rt.set_index("method")
    c.macros["RuntimeNQ"] = int(rt.n_queries.iloc[0])
    for m in e.index:
        k = MACRO_KEY.get(m, m)
        c.macros[f"Rt{k}"] = f"{e.loc[m, 'runtime_ms']:.2f}"
        c.macros[f"Exp{k}"] = f"{e.loc[m, 'expanded']:.0f}"
    if {"dijkstra_length", "astar_length", "td_dijkstra", "plan_once", "td_astar_lb", "td_astar_alt"} <= set(e.index):
        red = lambda a, b: f"{100 * (1 - e.loc[a, 'expanded'] / e.loc[b, 'expanded']):.1f}"  # noqa: E731
        c.macros.update(AstarReduction=red("astar_length", "dijkstra_length"), TdAstarReduction=red("plan_once", "td_dijkstra"),
                        LbReduction=red("td_astar_lb", "td_dijkstra"), AltReduction=red("td_astar_alt", "td_dijkstra"),
                        LbVsEuclid=red("td_astar_lb", "plan_once"), AltVsEuclid=red("td_astar_alt", "plan_once"),
                        LbPrep=f"{e.loc['td_astar_lb', 'prep_ms']:.2f}", AltPrep=f"{e.loc['td_astar_alt', 'prep_ms']:.2f}")
    if {"plan_once", "td_astar_numba"} <= set(e.index):
        c.macros["NumbaSpeedup"] = f"{e.loc['plan_once', 'runtime_ms'] / e.loc['td_astar_numba', 'runtime_ms']:.1f}"
    if {"replan", "replan_lazy", "replan_dstar"} <= set(e.index):
        c.macros["LazySpeedup"] = f"{e.loc['replan', 'runtime_ms'] / e.loc['replan_lazy', 'runtime_ms']:.1f}"
        c.macros["DstarSpeedup"] = f"{e.loc['replan', 'runtime_ms'] / e.loc['replan_dstar', 'runtime_ms']:.1f}"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col, lab in ((axes[0], "expanded", "Nút mở rộng (RL: số bước) TB / truy vấn"), (axes[1], "runtime_ms", "Thời gian đơn luồng TB (ms)")):
        ax.bar(rt.label, rt[col], color=[PALETTE.get(m, "#999") for m in rt.method])
        ax.set_yscale("log")
        ax.set_ylabel(lab + " (log)")
        ax.grid(axis="y", alpha=0.3)
        _rotate(ax)
    fig.tight_layout()
    c.save(fig, "expansions_runtime")


def fig_noise_sweep(c: Ctx) -> None:
    p = c.res / "noise_sweep.csv"
    if not p.exists():
        return
    d = add_regret(pd.read_csv(p).assign(query_id=lambda x: x.query_id + "_s" + x.sigma.astype(str)))
    g = d.groupby(["sigma", "method"]).agg(regret=("regret_pct", "mean")).reset_index()
    methods = [m for m in ["dijkstra_length", "plan_once", "plan_once_gnn", "replan", "replan_gnn"] if m in set(g.method)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in methods:
        s = g[g.method == m]
        ax.plot(s.sigma, s.regret, marker="o", label=METHOD_LABELS[m], color=PALETTE[m])
    nq = d.query_id.str.rsplit("_s", n=1).str[0].nunique()
    ax.set(xlabel="σ của nhiễu (không sự cố)", ylabel="Regret TB so với oracle (%)", title=f"Độ nhạy theo mức nhiễu ({nq} truy vấn kiểm tra)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    c.save(fig, "noise_sensitivity")
    piv = g.pivot(index="sigma", columns="method", values="regret")[methods].reset_index()
    c.table(piv, "tab_noise_sweep", {"sigma": "$\\sigma$"} | {m: METHOD_LABELS[m] for m in methods},
            "Regret trung bình (\\%) theo mức nhiễu $\\sigma$ khi không có sự cố.", digits=2)
    last = piv.iloc[-1]
    c.macros.update(NoiseNQ=nq, NoiseMaxSigma=f"{last.sigma:.1f}", NoisePoMax=_f(last.plan_once), NoiseRpMax=_f(last.replan),
                    NoiseDijMax=_f(last.dijkstra_length))
    if "plan_once_gnn" in piv:
        c.macros["NoisePoGnnMax"] = _f(last.plan_once_gnn)
    if "replan_gnn" in piv:
        c.macros["NoiseRpGnnMax"] = _f(last.replan_gnn)


def fig_saving_by_hour(c: Ctx) -> None:
    d = pd.read_csv(c.res / "saving_by_hour.csv")
    g = d.groupby("hour").agg(saving=("saving_pct", "mean"), q90=("saving_pct", lambda s: s.quantile(0.9)),
                              tt_fast=("tt_fastest", "mean"), tt_short=("tt_shortest", "mean"),
                              detour=("len_fastest_m", "mean"), short_len=("len_shortest_m", "mean")).reset_index()
    g["detour_pct"] = 100 * (g.detour / g.short_len - 1)
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.bar(g.hour, g.saving, color="#1f77b4", alpha=0.8, label="Tiết kiệm TB (%)")
    ax1.plot(g.hour, g.q90, color="#1f77b4", ls="--", marker=".", label="P90 tiết kiệm (%)")
    ax1.set(xlabel="Giờ xuất phát", ylabel="Tiết kiệm của tuyến nhanh nhất\nso với tuyến ngắn nhất (%)", xticks=range(0, 24, 2))
    ax2 = ax1.twinx()
    ax2.plot(g.hour, g.tt_short, color="#7f7f7f", marker="o", ms=3, label="TB tuyến ngắn nhất")
    ax2.plot(g.hour, g.tt_fast, color="#2ca02c", marker="o", ms=3, label="TB tuyến nhanh nhất")
    ax2.set_ylabel("Thời gian (phút, V_base)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Nhanh nhất (TD-A*) vs ngắn nhất (Dijkstra) theo giờ, {d.query_id.nunique()} cặp O–D kiểm tra")
    c.save(fig, "saving_by_hour")
    c.macros.update(SavingThree=f"{g.saving[3]:.1f}", SavingEight=f"{g.saving[8]:.1f}", SavingEighteen=f"{g.saving[18]:.1f}",
                    TTShortEight=f"{g.tt_short[8]:.2f}", TTShortThree=f"{g.tt_short[3]:.2f}", SavingNQ=d.query_id.nunique())
    sel = g[g.hour.isin([0, 3, 6, 7, 8, 9, 12, 14, 16, 17, 18, 19, 21])]
    c.table(sel, "tab_saving_by_hour", {"hour": "Giờ", "tt_short": "Ngắn nhất (phút)", "tt_fast": "Nhanh nhất (phút)",
            "saving": "Tiết kiệm TB (\\%)", "q90": "P90 (\\%)", "detour_pct": "Dài thêm (\\%)"},
            "Tiết kiệm thời gian của tuyến nhanh nhất so với tuyến ngắn nhất (trên vận tốc nền) theo giờ xuất phát.",
            digits={"hour": 0, "tt_short": 2, "tt_fast": 2, "saving": 2, "q90": 2, "detour_pct": 1})


def fig_rl_curves(c: Ctx, df: pd.DataFrame) -> None:
    sp = c.res / "rl" / "train_summary.json"
    if not sp.exists():
        return
    summ = load_json(sp)
    variants = [v["name"] for v in c.cfg["rl"]["variants"] if any(r["variant"] == v["name"] for r in summ["runs"])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for v in variants:
        hs = [pd.read_csv(Path(r["run_dir"]) / "eval_history.csv").assign(seed=r["seed"]) for r in summ["runs"] if r["variant"] == v]
        h = pd.concat(hs)
        if (h.timesteps == 0).any():  # điểm đánh giá ngay sau behavior cloning (trước PPO)
            c.macros[f"{MACRO_KEY.get(v, v)}AfterBc"] = f"{100 * h[h.timesteps == 0].success_rate.mean():.1f}"
        g = h.groupby("timesteps").agg(s_mean=("success_rate", "mean"), s_min=("success_rate", "min"), s_max=("success_rate", "max"),
                                       r_mean=("regret_pct", "mean")).reset_index()
        axes[0].plot(g.timesteps, 100 * g.s_mean, marker=".", color=PALETTE[v], label=METHOD_LABELS[v])
        axes[0].fill_between(g.timesteps, 100 * g.s_min, 100 * g.s_max, color=PALETTE[v], alpha=0.15)
        axes[1].plot(g.timesteps, g.r_mean, marker=".", color=PALETTE[v], label=METHOD_LABELS[v])
    from matplotlib.ticker import FuncFormatter

    for ax in axes:
        ax.set_xscale("symlog", linthresh=20000)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        ax.set_xlabel("Timesteps (symlog)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Tỉ lệ tới đích trên tập kiểm định (%) — TB, dải min–max 3 seed")
    axes[1].set_ylabel("Regret trên tập kiểm định (%) — chuyến tới đích")
    axes[1].set_yscale("symlog", linthresh=5)
    axes[0].legend(fontsize=8)
    fig.suptitle("Đường cong học theo biến thể (đánh giá định kỳ trên 100 truy vấn kiểm định)", fontsize=11)
    fig.tight_layout()
    c.save(fig, "rl_learning_curves")
    # bảng theo seed
    tp = c.res / "rl" / "rl_test_per_query.csv"
    rows = []
    t = pd.read_csv(tp) if tp.exists() else None
    oracle_tt = df[df.method == "oracle"].set_index("query_id").travel_time
    for r in summ["runs"]:
        row = {"variant": METHOD_LABELS.get(r["variant"], r["variant"]), "key": r["variant"], "seed": r["seed"],
               "timesteps": r["timesteps"], "best_val_success": 100 * r["best_val_success"], "best_at": r["best_at_timesteps"],
               "train_min": r["train_wall_s"] / 60, "selected": "x" if summ["selected_seed"].get(r["variant"]) == r["seed"] else "",
               "bc_acc": 100 * r["bc_accuracy"] if "bc_accuracy" in r else np.nan}
        if t is not None:
            sub = t[(t.variant == r["variant"]) & (t.seed == r["seed"]) & (t.incident_set == "main")]
            row["test_success"] = 100 * sub.reached.mean()
            reg = 100 * (sub.set_index("query_id").travel_time / oracle_tt.reindex(sub.query_id).values - 1)
            row["test_regret"] = float(np.nanmean(reg.values)) if sub.reached.any() else np.nan
        rows.append(row)
    tr = pd.DataFrame(rows).sort_values(["key", "seed"], key=lambda s: s.map({v: i for i, v in enumerate(variants)}) if s.name == "key" else s)
    c.table(tr, "tab_rl_seeds", {"variant": "Biến thể", "seed": "Seed", "timesteps": "Bước", "best_val_success": "Val tới đích (\\%)",
            "best_at": "Tốt nhất tại", "test_success": "Test tới đích (\\%)", "test_regret": "Test regret (\\%)",
            "train_min": "Huấn luyện (phút)", "selected": "Chọn"},
            "Kết quả học tăng cường theo từng seed (checkpoint tốt nhất theo tập kiểm định; cột Chọn đánh dấu seed dùng cho bảng chính).",
            digits={"seed": 0, "timesteps": 0, "best_at": 0, "best_val_success": 1, "test_success": 1, "test_regret": 2, "train_min": 1},
            align="lrrrrrrrc")
    agg = tr.groupby("key").agg(succ_mean=("test_success", "mean"), succ_std=("test_success", lambda s: s.std(ddof=0)),
                                reg_mean=("test_regret", "mean"), train=("train_min", "mean"), bc=("bc_acc", "mean"))
    for v, r in agg.iterrows():
        k = MACRO_KEY.get(v, v)
        c.macros[f"{k}SeedSucc"] = _f(r.succ_mean, 1)
        c.macros[f"{k}SeedSuccSd"] = _f(r.succ_std, 1)
        c.macros[f"{k}SeedRegret"] = _f(r.reg_mean)
        c.macros[f"{k}TrainMin"] = _f(r.train, 0)
        if np.isfinite(r.bc):
            c.macros[f"{k}BcAcc"] = _f(r.bc, 1)
    c.macros["RlNSeeds"] = len(c.cfg["rl"]["seeds"])


def fig_route_example(c: Ctx, df: pd.DataFrame) -> None:
    from matplotlib.collections import LineCollection

    from src.graph_data import RoadGraph
    from src.scenarios import load_split

    graph = RoadGraph.from_config(c.cfg)
    test = {q["id"]: q for q in load_split(c.cfg, "test")}
    methods = _present(df, ["dijkstra_length", "plan_once", "replan", "replan_gnn", "rl_tree_bc", "rl_hybrid", "oracle"])
    piv = df[df.method.isin(methods)].pivot_table(index="query_id", columns="method", values="travel_time")
    cand = [q for q in piv.dropna(subset=["plan_once", "replan"]).index if test[q]["incidents"]] or list(piv.index)
    score = (piv.loc[cand, "plan_once"] - piv.loc[cand, "replan"]) / piv.loc[cand, "oracle"]
    qid = score.idxmax()
    q = test[qid]
    segs = [cc[:, ::-1] for cc in graph.edge_coords]
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.add_collection(LineCollection(segs, colors="#d9d9d9", linewidths=0.4))
    widths = {"dijkstra_length": 7.0, "plan_once": 5.5, "replan": 4.2, "replan_gnn": 3.4, "rl_tree_bc": 2.6, "rl_hybrid": 2.0, "oracle": 1.2}
    sel = df[(df.query_id == qid) & df.method.isin(methods)].set_index("method")
    for m in methods:
        es = sel.loc[m, "edges"] if m in sel.index else ""
        if not isinstance(es, str) or not es.strip():
            continue
        tt = sel.loc[m, "travel_time"]
        lab = f"{METHOD_LABELS[m]}: {tt:.1f} phút" if np.isfinite(tt) else f"{METHOD_LABELS[m]}: không tới đích"
        ax.add_collection(LineCollection([segs[int(e)] for e in es.split()], colors=[PALETTE[m]], linewidths=widths[m],
                                         alpha=0.8, label=lab))
        if m in ("plan_once", "replan", "oracle", "rl_tree_bc", "rl_hybrid") and np.isfinite(tt):
            c.macros[f"Example{MACRO_KEY[m]}TT"] = f"{tt:.1f}"
    for inc in q["incidents"]:
        cc = graph.edge_coords[inc["edge"]].mean(axis=0)
        ax.plot(cc[1], cc[0], marker="X", color="magenta", ms=14, mec="k", zorder=5)
    ax.plot(graph.node_lon[q["origin"]], graph.node_lat[q["origin"]], "o", color="lime", ms=12, mec="k", zorder=6, label="Điểm đi")
    ax.plot(graph.node_lon[q["dest"]], graph.node_lat[q["dest"]], "s", color="red", ms=12, mec="k", zorder=6, label="Điểm đến")
    ax.plot([], [], "X", color="magenta", ms=10, mec="k", ls="", label="Sự cố")
    lat = [graph.node_lat[q["origin"]], graph.node_lat[q["dest"]]]
    lon = [graph.node_lon[q["origin"]], graph.node_lon[q["dest"]]]
    ax.set_xlim(min(lon) - 0.012, max(lon) + 0.012)
    ax.set_ylim(min(lat) - 0.012, max(lat) + 0.012)
    ax.set_aspect(1 / math.cos(math.radians(graph.lat0)))
    ax.ticklabel_format(useOffset=False)
    hh, mm = divmod(int(q["depart_min"]), 60)
    ax.set_title(f"Ví dụ {qid}: xuất phát {hh:02d}:{mm:02d}, σ={q['sigma']}, {len(q['incidents'])} sự cố", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.set(xlabel="Kinh độ", ylabel="Vĩ độ")
    c.save(fig, "route_example")
    c.macros["ExampleQuery"] = qid.replace("_", r"\_")
    c.macros["ExampleNInc"] = len(q["incidents"])


def fig_departure(c: Ctx) -> None:
    p = c.res / "departure" / "departure_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    d = pd.read_csv(c.res / "departure" / "departure_results.csv")
    pol_lab = {"go_now": "Đi ngay", "q_learning": "Q-learning", "oracle_best": "Giờ tối ưu (vét cạn)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    sp = s.assign(policy_label=s.policy.map(pol_lab), w=s.wait_weight.map(lambda w: f"w = {w}"))
    sns.barplot(data=sp, x="w", y="cost_mean", hue="policy_label", ax=axes[0], hue_order=list(pol_lab.values()))
    lo = sp.cost_mean.min()
    axes[0].set_ylim(lo * 0.97, sp.cost_mean.max() * 1.01)
    axes[0].set(ylabel="Chi phí TB = w·chờ + đi (phút)", xlabel="Trọng số thời gian chờ")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    for (pol, ls, mk) in (("oracle_best", "-", "o"), ("q_learning", "--", "x")):
        g = d[d.policy == pol].groupby(["wait_weight", "hour"]).k_wait.apply(lambda k: 100 * (k > 0).mean()).reset_index()
        for w, sub in g.groupby("wait_weight"):
            axes[1].plot(sub.hour, sub.k_wait, ls=ls, marker=mk, ms=3, label=f"{pol_lab[pol]}, w = {w}")
    axes[1].set(xlabel="Giờ yêu cầu chuyến đi", ylabel="Tỉ lệ truy vấn chọn chờ (%)", xticks=range(0, 24, 2))
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    c.save(fig, "departure_time")
    sp["policy_label"] = sp.policy.map(pol_lab)
    c.table(sp, "tab_departure", {"wait_weight": "w", "policy_label": "Chính sách", "cost_mean": "Chi phí TB (phút)",
            "travel_mean": "Đi TB (phút)", "total_mean": "Chờ+đi TB (phút)", "wait_share": "Tỉ lệ chờ", "regret_mean": "Regret (phút)"},
            f"Kết quả thí nghiệm chọn giờ xuất phát trên tập kiểm tra (bước chờ {c.cfg['departure']['wait_step_min']} phút, tối đa "
            f"{c.cfg['departure']['max_waits']} lần).", digits={"wait_weight": 2, "cost_mean": 2, "travel_mean": 2,
            "total_mean": 2, "wait_share": 3, "regret_mean": 3}, align="llrrrrr")
    wkey = {1.0: "WOne", 0.5: "WHalf", 0.25: "WLow"}
    for r in s.itertuples():
        key = {"go_now": "Go", "q_learning": "Q", "oracle_best": "Opt"}[r.policy] + wkey.get(float(r.wait_weight), "W")
        c.macros[f"Dep{key}Cost"] = f"{r.cost_mean:.2f}"
        c.macros[f"Dep{key}Wait"] = f"{100 * r.wait_share:.1f}"
        c.macros[f"Dep{key}Regret"] = f"{r.regret_mean:.3f}"
    c.macros["DepLowW"] = f"{min(c.cfg['departure']['wait_cost_weights'])}"
    c.macros["DepMinVisits"] = c.cfg["departure"]["min_visits"]
    js = load_json(c.res / "departure" / "departure_summary.json")
    c.macros["DepMaxSaving"] = f"{js['max_saving_by_waiting_min']:.1f}"


def forecast_tables(c: Ctx) -> None:
    p = c.res / "forecast" / "forecast_summary.json"
    if not p.exists():
        return
    js = load_json(p)
    a, b = js["val_after"], js["val_before"]
    df = pd.DataFrame([{"model": "Dự báo nền V_base (log-tỉ số = 0)", "unobs": a["zero_unobs"], "all": a["zero_all"]},
                       {"model": "GCN chưa huấn luyện", "unobs": b["gcn_unobs"], "all": b["gcn_all"]},
                       {"model": "GCN đã huấn luyện", "unobs": a["gcn_unobs"], "all": a["gcn_all"]}])
    c.table(df, "tab_forecast", {"model": "Mô hình", "unobs": "RMSE cạnh chưa quan sát", "all": "RMSE mọi cạnh"},
            f"Sai số dự báo log-tỉ số vận tốc (mốc $h$ và $h+1$) trên {c.cfg['forecast']['gcn']['val_fields']} trường kiểm định độc lập; "
            f"{js['n_sensors']} cảm biến ({100 * js['sensor_share']:.0f}\\% cạnh) + cạnh trong bán kính quan sát.",
            digits=4, align="lrr")
    c.macros.update(GcnRmseZero=f"{a['zero_unobs']:.3f}", GcnRmse=f"{a['gcn_unobs']:.3f}",
                    GcnImprove=f"{100 * (1 - a['gcn_unobs'] / a['zero_unobs']):.1f}", NSensors=js["n_sensors"],
                    GcnTrainMin=f"{js['train_s'] / 60:.1f}")
    h = pd.DataFrame(js["history"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h.step, h.zero_unobs, "--", color="gray", label="Dự báo nền (V_base)")
    ax.plot(h.step, h.gcn_unobs, marker="o", ms=3, label="GCN")
    ax.set(xlabel="Bước huấn luyện", ylabel="RMSE log-tỉ số (cạnh chưa quan sát)", title="Huấn luyện GCN dự báo ngắn hạn")
    ax.grid(alpha=0.3)
    ax.legend()
    c.save(fig, "gcn_training")


def risk_tables(c: Ctx) -> None:
    p = c.res / "risk_eval.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    piv = d.pivot_table(index="query_id", columns="route", values=["mean", "p90"])
    diff_p90 = piv["p90"]["risk_p90"] - piv["p90"]["plan_once"]
    diff_mean = piv["mean"]["risk_p90"] - piv["mean"]["plan_once"]
    same = d[d.route == "plan_once"].same.astype(bool)
    rows = [{"route": METHOD_LABELS[r], "mean": d[d.route == r]["mean"].mean(), "p90": d[d.route == r].p90.mean(),
             "std": d[d.route == r]["std"].mean()} for r in ("plan_once", "risk_p90")]
    c.table(pd.DataFrame(rows), "tab_risk", {"route": "Tuyến", "mean": "TB thời gian (phút)", "p90": "P90 (phút)", "std": "Độ lệch chuẩn"},
            f"Độ tin cậy tuyến trên {c.cfg['algorithms']['risk']['eval_samples']} kịch bản nhiễu độc lập cho mỗi truy vấn "
            f"({d.query_id.nunique()} truy vấn kiểm tra, không sự cố). Tuyến P90 khác tuyến plan-once ở {100 * (1 - same.mean()):.0f}\\% truy vấn.",
            digits=3, align="lrrr")
    changed = ~same.values
    c.macros.update(RiskEvalNQ=d.query_id.nunique(), RiskChangedPct=f"{100 * changed.mean():.1f}",
                    RiskDiffPn=f"{diff_p90.mean():.3f}", RiskDiffMean=f"{diff_mean.mean():.3f}",
                    RiskDiffPnChanged=f"{diff_p90[changed].mean():.3f}" if changed.any() else "--",
                    RiskDiffMeanChanged=f"{diff_mean[changed].mean():.3f}" if changed.any() else "--",
                    RiskBetterPnPct=f"{100 * (diff_p90[changed] < 0).mean():.1f}" if changed.any() else "--")


def multiagent_tables(c: Ctx) -> None:
    p = c.res / "multiagent.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    lab = {"no_interaction": "Không tương tác (lý tưởng)", "naive": "Dẫn đường giống nhau (naive)", "coordinated": "Phối hợp tuần tự"}
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for pol, sub in d.groupby("policy"):
        ax.plot(sub.n_vehicles, sub.tt_mean, marker="o", label=lab[pol])
    ax.set(xlabel="Số xe được dẫn đường (7:30–8:00)", ylabel="Thời gian đi TB (phút)", title="Ùn tắc dịch chuyển khi nhiều xe cùng làm theo chỉ dẫn")
    ax.grid(alpha=0.3)
    ax.legend()
    c.save(fig, "multiagent")
    piv = d.pivot(index="n_vehicles", columns="policy", values="tt_mean").reset_index()
    vc = d[d.policy != "no_interaction"].pivot(index="n_vehicles", columns="policy", values="max_vc").add_prefix("vc_").reset_index()
    piv = piv.merge(vc, on="n_vehicles")
    c.table(piv, "tab_multiagent", {"n_vehicles": "Số xe", "no_interaction": "Không tương tác", "naive": "Naive",
            "coordinated": "Phối hợp", "vc_naive": "V/C max (naive)", "vc_coordinated": "V/C max (phối hợp)"},
            "Thời gian đi trung bình (phút) khi nhiều xe được dẫn đường cùng lúc; V/C là tỉ số lớn nhất giữa lưu lượng dẫn đường và năng lực.",
            digits={"n_vehicles": 0, "no_interaction": 2, "naive": 2, "coordinated": 2, "vc_naive": 2, "vc_coordinated": 2})
    last = piv.iloc[-1]
    c.macros.update(MaVeh=int(last.n_vehicles), MaFree=f"{last.no_interaction:.2f}", MaNaive=f"{last.naive:.2f}",
                    MaCoord=f"{last.coordinated:.2f}", MaNaiveInc=f"{100 * (last.naive / last.no_interaction - 1):.1f}",
                    MaCoordGain=f"{100 * (1 - last.coordinated / last.naive):.1f}", MaVcNaive=f"{last.vc_naive:.1f}")


# ---------------------------------------------------------------------------
_MD_INT_COLS = {"n_vehicles", "n_pairs"}


def _md_cell(k: str, v: Any, digits: int) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        return f"{v:.0f}" if k in _MD_INT_COLS else f"{v:.{digits}f}"
    # markdown dùng dấu chấm thập phân như các cột số khác; bỏ ký hiệu LaTeX
    return str(v).replace("{,}", ".").replace("$<10^{-4}$", "< 1e-4")


def _md_table(df: pd.DataFrame, cols: dict[str, str], digits: int = 2) -> str:
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = [_md_cell(k, r[k], digits) for k in cols]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def results_markdown(c: Ctx) -> str:
    """Mục kết quả chính dạng Markdown, sinh từ CSV trong reports/tables (không gõ tay)."""
    m = c.macros
    cols = {"label": "Phương pháp", "success_pct": "Tới đích %", "tt_mean": "TB (phút)", "tt_p90": "P90",
            "regret_mean": "Regret TB %", "regret_p90": "Regret P90 %"}
    parts = [f"*Tự động sinh bởi `src/evaluation/plots.py` từ `results/` (chế độ {c.cfg.mode}). Không sửa tay.*\n",
             f"**Đồ thị:** {m['NNodes']} nút, {m['NEdges']} cạnh, {m['TotalKm']} km. **Tập test:** {m['NTest']} truy vấn.\n",
             "### Tìm kiếm / dự báo (tập chính)\n", _md_table(pd.read_csv(c.tab / "tab_main_classic.csv"), cols),
             "\n### RL (tập chính)\n", _md_table(pd.read_csv(c.tab / "tab_main_rl.csv"), cols)]
    for name, title, cl in (("tab_stress", "Tập stress (sự cố trên tuyến dự kiến)", cols),
                            ("tab_runtime_single", "Thời gian tính đơn luồng",
                             {"label": "Phương pháp", "expanded": "Nút/bước TB", "runtime_ms": "ms TB", "prep_ms": "Tiền xử lý ms"}),
                            ("tab_stat_tests", "So sánh ghép cặp", {"pair": "Cặp (A − B)", "n_pairs": "n", "mean_diff_min": "Hiệu TB (phút)",
                                                                     "ci_lo": "CI thấp", "ci_hi": "CI cao", "p_str": "p"}),
                            ("tab_forecast", "Dự báo GCN", {"model": "Mô hình", "unobs": "RMSE chưa quan sát", "all": "RMSE mọi cạnh"}),
                            ("tab_risk", "Định tuyến P90", {"route": "Tuyến", "mean": "TB", "p90": "P90", "std": "Std"}),
                            ("tab_multiagent", "Nhiều xe", {"n_vehicles": "Số xe", "no_interaction": "Không tương tác", "naive": "Naive", "coordinated": "Phối hợp"}),
                            ("tab_departure", "Chọn giờ xuất phát", {"wait_weight": "w", "policy_label": "Chính sách", "cost_mean": "Chi phí TB",
                                                                     "wait_share": "Tỉ lệ chờ", "regret_mean": "Regret"})):
        p = c.tab / f"{name}.csv"
        if p.exists():
            parts += [f"\n### {title}\n", _md_table(pd.read_csv(p), cl, 3 if name in ("tab_stat_tests", "tab_forecast", "tab_departure") else 2)]
    parts.append("\nHình: `reports/figures/*.png|pdf`; bảng LaTeX: `reports/tables/*.tex`; báo cáo: `reports/paper/main.pdf`.\n")
    return "\n".join(parts)


def update_features(c: Ctx) -> None:
    from src.config import PROJECT_ROOT

    p = PROJECT_ROOT / "FEATURES.md"
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    if start not in txt or end not in txt:
        log.warning("FEATURES.md thiếu marker kết quả — bỏ qua")
        return
    pre, rest = txt.split(start, 1)
    _, post = rest.split(end, 1)
    p.write_text(pre + start + "\n" + results_markdown(c) + end + post, encoding="utf-8")
    log.info("Đã cập nhật mục kết quả trong FEATURES.md")


def run(cfg: Config) -> dict[str, Any]:
    c = Ctx(cfg)
    graph_tables(c)
    config_table(c)
    df = main_results(c)
    stress_results(c)
    breakdown_tables(c, df)
    route_overlap(c, df)
    fig_boxplots(c, df)
    fig_regret_by_hour(c, df)
    fig_breakdowns(c, df)
    runtime_tables(c, df)
    fig_noise_sweep(c)
    fig_saving_by_hour(c)
    fig_rl_curves(c, df)
    fig_route_example(c, df)
    fig_departure(c)
    forecast_tables(c)
    risk_tables(c)
    multiagent_tables(c)
    write_macros(c.macros, c.tab / "macros.tex")
    (c.tab / "results_summary.md").write_text(results_markdown(c), encoding="utf-8")
    if cfg.mode == "full":
        update_features(c)
    log.info("Đã ghi %d macro, hình vào %s, bảng vào %s", len(c.macros), c.fig, c.tab)
    return c.macros


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg)


if __name__ == "__main__":
    main()
