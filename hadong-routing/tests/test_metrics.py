"""Test giai đoạn 7: chỉ số, bootstrap, kiểm định, xuất LaTeX."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import add_regret, bootstrap_mean_diff, paired_tests, summarize, to_latex_table, write_macros


def _toy() -> pd.DataFrame:
    rows = []
    for i in range(30):
        base = 10 + i % 7
        for m, f in (("oracle", 1.0), ("plan_once", 1.1), ("replan", 1.03)):
            rows.append({"query_id": f"q{i}", "method": m, "reached": True, "travel_time": base * f,
                         "length_m": 3000.0, "runtime_s": 0.01, "expanded": 100})
    return pd.DataFrame(rows)


def test_regret_and_summary():
    df = add_regret(_toy())
    s = summarize(df).set_index("method")
    assert s.loc["oracle", "regret_mean"] == pytest.approx(0.0)
    assert s.loc["plan_once", "regret_mean"] == pytest.approx(10.0)
    assert s.loc["replan", "success_pct"] == 100.0


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    a = rng.normal(5, 1, 200)
    b = rng.normal(4, 1, 200)
    m, lo, hi = bootstrap_mean_diff(a, b, 2000, 0.95, 1)
    assert lo < m < hi and lo > 0.5


def test_paired_tests_significant():
    t = paired_tests(add_regret(_toy()), [["plan_once", "replan"]], 1000, 0.95, 0)
    assert t.iloc[0].mean_diff_min > 0 and t.iloc[0].p_value < 0.01


def test_latex_outputs(tmp_path):
    df = pd.DataFrame({"a": ["x_y"], "b": [1.2345]})
    to_latex_table(df, {"a": "A", "b": "B"}, tmp_path / "t.tex", "cap", "tab:t")
    txt = (tmp_path / "t.tex").read_text(encoding="utf-8")
    assert r"x\_y" in txt and "1{,}23" in txt and r"\toprule" in txt  # dấu thập phân kiểu Việt Nam
    write_macros({"Foo": 1.25, "Bar": "[-1.74; 0.07]", "Baz": r"test\_0169"}, tmp_path / "m.tex")
    m = (tmp_path / "m.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\Foo}{1{,}2}" in m or r"\newcommand{\Foo}{1{,}3}" in m
    assert r"\newcommand{\Bar}{[-1{,}74; 0{,}07]}" in m and r"\newcommand{\Baz}{test\_0169}" in m
    with pytest.raises(ValueError):
        write_macros({"Bad1": 1}, tmp_path / "x.tex")
