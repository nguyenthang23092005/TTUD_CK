"""Chỉ số đánh giá, bootstrap CI, kiểm định Wilcoxon và xuất bảng LaTeX/CSV."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

METHOD_LABELS: dict[str, str] = {
    "dijkstra_length": "Dijkstra (ngắn nhất)",
    "astar_length": "A* (ngắn nhất)",
    "freeflow": "A* (thời gian tự do)",
    "td_dijkstra": "TD-Dijkstra (V_base)",
    "td_astar_lb": "TD-A* + heuristic LB",
    "td_astar_alt": "TD-A* + ALT",
    "td_astar_numba": "TD-A* (Numba)",
    "plan_once": "Plan-once TD-A*",
    "plan_once_sensor": "Plan-once + cảm biến",
    "plan_once_gnn": "Plan-once + GCN",
    "risk_p90": "Định tuyến P90",
    "replan": "Replan TD-A*",
    "replan_lazy": "Replan lười",
    "replan_dstar": "Replan D* Lite",
    "replan_gnn": "Replan + GCN",
    "greedy_local": "Tham lam cục bộ",
    "rl": "RL cục bộ",
    "rl_ctg": "RL + ctg (quan sát)",
    "rl_shape": "RL + định hình ctg",
    "rl_tree": "RL cây-GNN",
    "rl_lstm": "RL LSTM",
    "rl_tree_bc": "RL cây-GNN + BC",
    "rl_hybrid": "RL lai (β + TD-A*)",
    "oracle": "Oracle TD-A*",
}
CLASSIC_METHODS = ["dijkstra_length", "freeflow", "plan_once", "plan_once_sensor", "plan_once_gnn", "risk_p90",
                   "replan", "replan_lazy", "replan_dstar", "replan_gnn", "oracle"]
RL_METHODS = ["greedy_local", "rl", "rl_ctg", "rl_shape", "rl_tree", "rl_lstm", "rl_tree_bc", "rl_hybrid"]
TRIP_METHODS = CLASSIC_METHODS[:-1] + RL_METHODS + ["oracle"]
MACRO_KEY: dict[str, str] = {
    "dijkstra_length": "Dij", "freeflow": "Ff", "plan_once": "Po", "plan_once_sensor": "PoSen", "plan_once_gnn": "PoGnn",
    "risk_p90": "Risk", "replan": "Rp", "replan_lazy": "RpLazy", "replan_dstar": "RpDstar", "replan_gnn": "RpGnn",
    "greedy_local": "Gr", "rl": "Rl", "rl_ctg": "Rlc", "rl_shape": "RlShape", "rl_tree": "RlTree", "rl_lstm": "RlLstm",
    "rl_tree_bc": "RlTreeBc", "rl_hybrid": "RlHyb", "oracle": "Or", "astar_length": "Astar", "td_dijkstra": "TdDij",
    "td_astar_lb": "TdLb", "td_astar_alt": "TdAlt", "td_astar_numba": "TdNumba",
}


def add_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột regret_pct = 100·(tt / tt_oracle − 1) theo từng truy vấn."""
    o = df[df.method == "oracle"].set_index("query_id")["travel_time"]
    df = df.copy()
    df["oracle_tt"] = df["query_id"].map(o)
    df["regret_pct"] = 100.0 * (df["travel_time"] / df["oracle_tt"] - 1.0)
    return df


def summarize(df: pd.DataFrame, methods: Sequence[str] | None = None) -> pd.DataFrame:
    """Bảng tổng hợp theo phương pháp (thời gian & regret chỉ tính trên các chuyến tới đích)."""
    rows = []
    for m in methods or [m for m in METHOD_LABELS if m in set(df.method)]:
        d = df[df.method == m]
        if d.empty:
            continue
        ok = d[d.reached]
        rows.append({
            "method": m, "label": METHOD_LABELS.get(m, m), "n": len(d), "success_pct": 100 * d.reached.mean(),
            "tt_mean": ok.travel_time.mean(), "tt_median": ok.travel_time.median(),
            "tt_p90": ok.travel_time.quantile(0.9), "regret_mean": ok.regret_pct.mean(),
            "regret_median": ok.regret_pct.median(), "regret_p90": ok.regret_pct.quantile(0.9),
            "length_km": ok.length_m.mean() / 1000, "runtime_ms": 1000 * d.runtime_s.mean(),
            "expanded": d.expanded.mean(),
        })
    return pd.DataFrame(rows)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int, level: float, seed: int) -> tuple[float, float, float]:
    """CI bootstrap (percentile) cho trung bình của hiệu ghép cặp a − b."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_tests(df: pd.DataFrame, pairs: Sequence[Sequence[str]], n_boot: int, level: float, seed: int) -> pd.DataFrame:
    """So sánh từng cặp phương pháp trên các truy vấn mà cả hai cùng tới đích."""
    piv = df.pivot_table(index="query_id", columns="method", values="travel_time")
    rows = []
    for a, b in pairs:
        if a not in piv or b not in piv:
            continue
        sub = piv[[a, b]].dropna()
        if len(sub) < 2:
            continue
        mean, lo, hi = bootstrap_mean_diff(sub[a].to_numpy(), sub[b].to_numpy(), n_boot, level, seed)
        diff = sub[a] - sub[b]
        if np.allclose(diff, 0):
            w_stat, p = float("nan"), 1.0
        else:
            res = stats.wilcoxon(sub[a], sub[b], zero_method="wilcox")
            w_stat, p = float(res.statistic), float(res.pvalue)
        rows.append({"A": a, "B": b, "n_pairs": len(sub), "mean_diff_min": mean, "ci_lo": lo, "ci_hi": hi,
                     "mean_diff_pct": 100 * float((diff / sub[b]).mean()), "A_better_pct": 100 * float((diff < -1e-9).mean()),
                     "wilcoxon_W": w_stat, "p_value": p})
    return pd.DataFrame(rows)


def breakdown(df: pd.DataFrame, by: str, methods: Sequence[str]) -> pd.DataFrame:
    """Regret TB, thời gian TB và tỉ lệ tới đích theo nhóm `by`."""
    d = df[df.method.isin(methods)]
    g = d.groupby([by, "method"])
    out = g.agg(n=("query_id", "size"), success_pct=("reached", lambda s: 100 * s.mean())).reset_index()
    ok = d[d.reached].groupby([by, "method"]).agg(tt_mean=("travel_time", "mean"), regret_mean=("regret_pct", "mean")).reset_index()
    return out.merge(ok, on=[by, "method"], how="left")


# ---------------------------------------------------------------------------
# Xuất LaTeX
# ---------------------------------------------------------------------------
def _fmt(v: Any, digits: int) -> str:
    if isinstance(v, str):
        return v.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    return vn_decimal(f"{float(v):.{digits}f}")


_NUMERIC = re.compile(r"^[\[\]\-+\d.;<>e ]+$")


def vn_decimal(s: str) -> str:
    """Dấu thập phân kiểu Việt Nam cho LaTeX: 12.34 -> 12{,}34 (chỉ áp dụng cho chuỗi thuần số)."""
    return s.replace(".", "{,}") if _NUMERIC.match(s) and any(ch.isdigit() for ch in s) else s


def to_latex_table(df: pd.DataFrame, columns: dict[str, str], path: Path, caption: str, label: str,
                   digits: int | dict[str, int] = 2, align: str | None = None, note: str | None = None) -> None:
    """Ghi bảng booktabs (môi trường table) có thể \\input trực tiếp vào báo cáo."""
    cols = list(columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    dig = (lambda c: digits.get(c, 2)) if isinstance(digits, dict) else (lambda c: digits)
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(columns[c] for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(_fmt(r[c], dig(c)) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines.append(rf"\par\smallskip\footnotesize {note}")
    lines.append(r"\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_macros(values: dict[str, Any], path: Path, digits: int = 1) -> None:
    """Ghi các \\newcommand chứa số liệu (để văn bản báo cáo không gõ tay con số)."""
    lines = ["% Tự động sinh bởi src/evaluation/plots.py — KHÔNG sửa tay"]
    for k, v in values.items():
        if not k.isalpha():
            raise ValueError(f"Tên macro LaTeX chỉ được chứa chữ cái: {k}")
        if isinstance(v, (float, np.floating)):
            s = "--" if not np.isfinite(v) else f"{v:.{digits}f}"
        else:
            s = str(v)
        s = vn_decimal(s)
        lines.append(rf"\newcommand{{\{k}}}{{{s}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
