"""Giai đoạn 8 — giao diện Streamlit: tìm đường động trên mạng Hà Đông.

Chạy (từ thư mục hadong-routing/):  streamlit run src/app/streamlit_app.py
Tham số URL: ?mode=full&o=100&d=2500&h=8&all=1  (mở sẵn cặp O–D, bật mọi phương pháp)
Mặc định chạy theo giờ thực (GMT+7, tự cập nhật mỗi phút); có tham số h thì dùng giờ thủ công.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import folium  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from branca.element import MacroElement, Template  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from src.algorithms.dijkstra import shortest_by_length  # noqa: E402
from src.algorithms.dstar_lite import replan_dstar  # noqa: E402
from src.algorithms.replan import replan  # noqa: E402
from src.algorithms.risk import ScenarioBank, path_times, risk_route  # noqa: E402
from src.algorithms.td_astar import td_astar  # noqa: E402
from src.algorithms.yen import yen_k_shortest  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation.metrics import METHOD_LABELS  # noqa: E402
from src.graph_data import RoadGraph  # noqa: E402
from src.speed_model import Incident, base_model, get_noise_model, load_vbase, real_V  # noqa: E402
from src.utils import load_json  # noqa: E402

st.set_page_config(page_title="Tìm đường động Hà Đông", layout="wide")

CLASSIC = {"Dijkstra ngắn nhất": "#555555", "TD-A* nhanh nhất": "#1f77b4", "Replan TD-A*": "#2ca02c",
           "Replan D* Lite": "#17becf", "Định tuyến P90": "#8c564b"}
RL_COLORS = ["#d62728", "#ff7f0e", "#9467bd", "#e377c2", "#bcbd22", "#7f7f7f", "#1a1a1a"]
CLASSES = [(0.8, "#1a9850", "> 0.80 (thông thoáng)"), (0.65, "#91cf60", "0.65–0.80"), (0.5, "#fee08b", "0.50–0.65"),
           (0.35, "#fc8d59", "0.35–0.50"), (-1, "#d73027", "< 0.35 (ùn tắc)")]
NOISE_SEED = 12345
TZ = ZoneInfo("Asia/Ho_Chi_Minh")
# độ rộng nét pydeck theo MÉT cho trunk/primary, secondary/tertiary, residential (kèm chặn dưới theo pixel)
DECK_WIDTH_M = (14.0, 8.0, 4.0)


def now_vn() -> tuple[int, int]:
    """Giờ, phút hiện tại theo giờ Việt Nam (chế độ thời gian thực)."""
    now = dt.datetime.now(TZ)
    return now.hour, now.minute


def speeds_at(V: np.ndarray, t0: float) -> np.ndarray:
    """Vận tốc mọi cạnh tại phút t0, nội suy tuyến tính giữa hai khung giờ (giống TravelTimeModel.base_speed_mpm)."""
    tm = float(t0) % 1440.0
    h = int(tm // 60)
    f = (tm - 60.0 * h) / 60.0
    return V[:, h] * (1.0 - f) + V[:, (h + 1) % 24] * f


# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp đồ thị và vận tốc...")
def load_data(mode: str) -> dict[str, Any]:
    cfg = load_config(mode)
    graph = RoadGraph.from_config(cfg)
    V = load_vbase(cfg)
    _ = graph.edge_coords  # parse hình học một lần
    return {"cfg": cfg, "graph": graph, "V": V}


@st.cache_resource(show_spinner="Đang nạp chính sách RL...")
def load_rl(mode: str) -> dict[str, Any]:
    """Checkpoint RL tốt nhất theo val cho từng biến thể; {} nếu chưa huấn luyện."""
    cfg = load_config(mode)
    p = cfg.path("results_dir") / "rl" / "train_summary.json"
    if not p.exists():
        return {}
    from src.rl.evaluate_rl import load_model

    summ = load_json(p)
    out = {}
    for var in cfg["rl"]["variants"]:
        seed = summ["selected_seed"].get(var["name"])
        run = next((r for r in summ["runs"] if r["variant"] == var["name"] and r["seed"] == seed), None)
        if run and Path(run["best_model_path"]).exists():
            out[var["name"]] = (load_model(Path(run["best_model_path"]), run["algo"]), run["algo"], var)
    return out


@st.cache_resource(show_spinner="Đang dựng ngân hàng kịch bản nhiễu (dải tin cậy)...")
def load_bank(mode: str, sigma: float) -> np.ndarray:
    d = load_data(mode)
    n = int(d["cfg"]["algorithms"]["risk"]["n_samples"])
    return ScenarioBank(d["V"], d["graph"].length, get_noise_model(d["graph"], d["cfg"]), [sigma], n, 777).bank(sigma)


def ratio_class(r: float) -> int:
    return next(i for i, (th, _, _) in enumerate(CLASSES) if r > th)


@st.cache_data(show_spinner=False)
def speed_geojson(mode: str, t0: int, groups: tuple[str, ...]) -> dict[str, Any]:
    """Lớp vận tốc GỘP theo 5 lớp màu: 5 MultiLineString thay cho hàng nghìn đường riêng (nhẹ hơn nhiều)."""
    d = load_data(mode)
    g = d["graph"]
    ratio = speeds_at(d["V"], t0) / g.v_free
    gid = [g.group_names.index(x) for x in groups]
    minor = g.group_names.index("residential") if "residential" in g.group_names else -1
    # gộp theo (lớp màu, đường nhỏ hay không) để đường nhỏ vẽ mảnh hơn trục chính
    lines: dict[tuple[int, bool], list[list[list[float]]]] = {}
    for e in np.flatnonzero(np.isin(g.group_id, gid)):
        key = (ratio_class(float(ratio[e])), bool(g.group_id[e] == minor))
        lines.setdefault(key, []).append(g.edge_coords[e][:, ::-1].round(5).tolist())
    feats = [{"type": "Feature", "properties": {"color": CLASSES[i][1], "label": CLASSES[i][2], "n": len(ls),
                                                "weight": 1.5 if is_minor else 3},
              "geometry": {"type": "MultiLineString", "coordinates": ls}}
             for (i, is_minor), ls in sorted(lines.items(), key=lambda kv: not kv[0][1])]  # đường nhỏ vẽ trước (nằm dưới)
    return {"type": "FeatureCollection", "features": feats}


@st.cache_data(show_spinner=False)
def pydeck_paths(mode: str, t0: int) -> pd.DataFrame:
    d = load_data(mode)
    g = d["graph"]
    v = speeds_at(d["V"], t0)
    ratio = v / g.v_free
    rgb = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for _, c, _ in CLASSES]
    wt, ws, wr = DECK_WIDTH_M
    return pd.DataFrame({"path": [g.edge_coords[e][:, ::-1].round(6).tolist() for e in range(g.n_edges)],
                         "color": [list(rgb[ratio_class(float(r))]) for r in ratio],
                         "width": np.where(g.group_id <= 1, wt, np.where(g.group_id <= 3, ws, wr)).tolist(),
                         "name": g.edge_names, "v": v.round(1), "ratio": ratio.round(2)})


def legend_element() -> MacroElement:
    rows = "".join(f"<div><span style='background:{c};width:18px;height:6px;display:inline-block;margin-right:6px'></span>{lab}</div>"
                   for _, c, lab in CLASSES)
    html = ("{% macro html(this, kwargs) %}<div style='position:absolute;bottom:24px;left:12px;z-index:9999;background:white;"
            "padding:6px 8px;border:1px solid #999;font-size:12px;border-radius:4px'><b>v / v_free</b>" + rows + "</div>{% endmacro %}")
    el = MacroElement()
    el._template = Template(html)
    return el


def rl_route(key: str, models: dict[str, Any], d: dict[str, Any], q: dict[str, Any]) -> dict[str, Any]:
    from src.rl.evaluate_rl import rollout
    from src.rl.train import make_env

    model, algo, var = models[key]
    env = make_env(d["cfg"], var, d["graph"], d["V"], [q], 0, eval_mode=True)
    return rollout(model, algo, env, q)


def band(V_bank: np.ndarray, g: RoadGraph, edges: list[int], t0: float) -> tuple[float, float, float]:
    if not edges:
        return (np.nan, np.nan, np.nan)
    tt = path_times(V_bank, g.length, edges, t0)
    return tuple(float(x) for x in np.quantile(tt, [0.1, 0.5, 0.9]))


@st.fragment(run_every=20)
def live_clock(shown: tuple[int, int]) -> None:
    """Chế độ giờ thực: khi sang phút mới thì chạy lại toàn bộ trang để cập nhật ùn tắc và tuyến."""
    if now_vn() != shown:
        st.rerun(scope="app")


# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🚦 Tìm đường tối ưu trong mạng giao thông động — Hà Đông")
    st.caption("Vận tốc là MÔ PHỎNG (không phải đo thực). Nhấp bản đồ để chọn điểm đi/đến hoặc vị trí sự cố.")
    ss = st.session_state
    for k, v in (("origin", None), ("dest", None), ("incidents", []), ("last_click", None)):
        ss.setdefault(k, v)
    qp = st.query_params
    if "qp_done" not in ss:
        ss.qp_done = True
        if "o" in qp and "d" in qp:
            ss.origin, ss.dest = int(qp["o"]), int(qp["d"])

    with st.sidebar:
        mode = st.selectbox("Bộ dữ liệu/kết quả", ["full", "smoke"], index=1 if qp.get("mode") == "smoke" else 0)
        d = load_data(mode)
        cfg, g, V = d["cfg"], d["graph"], d["V"]
        models = load_rl(mode)
        st.subheader("Thời gian")
        live = st.toggle("Theo giờ thực (real time)", value="h" not in qp,
                         help="Lấy giờ hiện tại của máy (giờ Việt Nam), tự cập nhật mỗi phút: "
                              "mức ùn tắc và tuyến đường tính theo đúng thời điểm đang chạy chương trình.")
        if live:
            hour, minute = now_vn()
            st.markdown(f"🕒 Bây giờ: **{hour:02d}:{minute:02d}** (GMT+7)")
            st.slider("Giờ xuất phát", 0, 23, hour, disabled=True)
            st.slider("Phút", 0, 59, minute, disabled=True)
        else:
            hour = st.slider("Giờ xuất phát", 0, 23, int(qp.get("h", cfg["app"]["default_hour"])))
            minute = st.slider("Phút", 0, 59, 0, step=5)
        t0 = 60.0 * hour + minute
        st.subheader("Phương pháp")
        rl_opts = {METHOD_LABELS.get(k, k): k for k in models}
        options = list(CLASSIC) + list(rl_opts)
        default = options if qp.get("all") == "1" else options[:3]
        methods = st.multiselect("Hiển thị", options, default=default)
        if not models:
            st.info("Chưa có checkpoint RL — chạy `python run_all.py --mode full` để huấn luyện.")
        k_alt = st.slider("Số tuyến thay thế (Yen)", 0, 5, int(cfg["algorithms"]["yen_k"]))
        sigma = st.select_slider("Nhiễu thực tế σ (môi trường mô phỏng)", options=cfg["scenarios"]["sigma_levels"],
                                 value=cfg["scenarios"]["sigma_levels"][1])
        show_band = st.checkbox("Dải tin cậy P10–P90 (ngân hàng kịch bản)", value=True)
        show_speed = st.checkbox("Lớp vận tốc/ùn tắc theo giờ", value=True)
        show_minor = st.checkbox("Gồm cả đường nhỏ, ngõ (residential)", value=True, disabled=not show_speed,
                                 help="Đường nhỏ chiếm ~80% số cạnh; tắt đi để bản đồ nhẹ và dễ nhìn các trục chính hơn.")
        speed_groups = tuple(cfg["app"]["show_speed_layer_groups"])
        if show_minor:
            speed_groups = tuple(g.group_names)
        st.subheader("Chế độ nhấp chuột")
        click_mode = st.radio("Nhấp bản đồ để", ["Đặt điểm đi", "Đặt điểm đến", "Thêm sự cố"])
        sev = st.slider("Mức sự cố (hệ số vận tốc)", 0.1, 0.9, 0.2, 0.05)
        dur = st.slider("Thời lượng sự cố (phút)", 15, 180, 60, 15)
        if st.button("Xoá mọi sự cố"):
            ss.incidents = []
        with st.expander("Nhập toạ độ thủ công"):
            olat = st.number_input("Vĩ độ điểm đi", value=20.9715, format="%.6f")
            olon = st.number_input("Kinh độ điểm đi", value=105.7787, format="%.6f")
            dlat = st.number_input("Vĩ độ điểm đến", value=20.9890, format="%.6f")
            dlon = st.number_input("Kinh độ điểm đến", value=105.7930, format="%.6f")
            if st.button("Dùng toạ độ này"):
                ss.origin, ss.dest = g.nearest_node(olat, olon), g.nearest_node(dlat, dlon)

    if live:
        live_clock((hour, minute))
    incs = [Incident(**i) for i in ss.incidents]
    forecast = base_model(g, V, cfg, incs)
    V_bank = load_bank(mode, float(sigma)) if show_band else None
    rows, routes, alts = [], {}, []
    o, dst = ss.origin, ss.dest
    rp = cfg["algorithms"]["replan"]
    if o is not None and dst is not None and o != dst:
        real = base_model(g, real_V(g, V, float(sigma), NOISE_SEED, cfg), cfg, incs)

        def add(label: str, edges: list[int], planned: float, real_tt: float, expanded: Any, rt: float,
                banded: bool = True) -> None:
            routes[label] = edges
            row = {"Phương pháp": label, "Dự kiến (phút)": planned, "Thực (phút)": real_tt,
                   "Độ dài (km)": g.path_length(edges) / 1000, "Nút mở rộng/bước": expanded, "Tính (ms)": 1000 * rt}
            if V_bank is not None and banded:  # tuyến dở dang (RL không tới đích) không có dải tin cậy
                p10, p50, p90 = band(V_bank, g, edges, t0)
                row.update({"P10": p10, "P50": p50, "P90": p90})
            rows.append(row)

        sp = shortest_by_length(g, o, dst)
        fast = td_astar(g, o, dst, t0, forecast)
        if "Dijkstra ngắn nhất" in methods:
            add("Dijkstra ngắn nhất", sp.edges, forecast.path_travel_time(sp.edges, t0), real.path_travel_time(sp.edges, t0), sp.expanded, sp.runtime_s)
        if "TD-A* nhanh nhất" in methods:
            add("TD-A* nhanh nhất", fast.edges, fast.cost, real.path_travel_time(fast.edges, t0), fast.expanded, fast.runtime_s)
        if "Replan TD-A*" in methods:
            r = replan(g, o, dst, t0, forecast, real, rp["observation_radius_m"], rp["remember_observations"])
            add("Replan TD-A*", r.edges, r.planned_time, r.travel_time, r.expanded, r.runtime_s)
        if "Replan D* Lite" in methods:
            r = replan_dstar(g, o, dst, t0, forecast, real, rp["observation_radius_m"], cfg["algorithms"]["dstar"]["refresh_min"])
            add("Replan D* Lite", r.edges, r.planned_time, r.travel_time, r.expanded, r.runtime_s)
        if "Định tuyến P90" in methods:
            bank = V_bank if V_bank is not None else load_bank(mode, float(sigma))
            r = risk_route(g, o, dst, t0, forecast, real, bank, int(cfg["algorithms"]["risk"]["k_candidates"]))
            add("Định tuyến P90", r.edges, r.extra["planned_quantile"], r.travel_time, r.expanded, r.runtime_s)
        q = {"id": "app", "origin": o, "dest": dst, "depart_min": t0, "sigma": float(sigma), "noise_seed": NOISE_SEED,
             "incidents": ss.incidents, "ref_hops": len(fast.edges), "ref_time_base_min": fast.cost,
             "od_dist_m": g.straight_dist(o, dst)}
        for lab, key in rl_opts.items():
            if lab in methods:
                res = rl_route(key, models, d, q)
                add(lab + ("" if res["reached"] else " — KHÔNG tới đích"), res["edges"], np.nan, res["travel_time"],
                    res["steps"], res["runtime_s"], banded=res["reached"])
        if k_alt > 0:
            w_snap = g.length / np.maximum(forecast.base_speed_mpm(np.arange(g.n_edges), t0), 1e-6)
            for i, p in enumerate(yen_k_shortest(g, o, dst, w_snap, k_alt)):
                alts.append(p.edges)
                rows.append({"Phương pháp": f"Tuyến thay thế #{i + 1} (Yen)", "Dự kiến (phút)": forecast.path_travel_time(p.edges, t0),
                             "Thực (phút)": real.path_travel_time(p.edges, t0), "Độ dài (km)": g.path_length(p.edges) / 1000})

    tab_map, tab_net = st.tabs(["Bản đồ tuyến (folium)", "Toàn mạng (pydeck, nhanh)"])
    with tab_map:
        col_map, col_info = st.columns([3, 2])
        with col_map:
            fmap = folium.Map(location=[float(g.node_lat.mean()), float(g.node_lon.mean())], zoom_start=14, tiles="OpenStreetMap")
            if show_speed:
                folium.GeoJson(speed_geojson(mode, int(t0), speed_groups), name="Vận tốc",
                               style_function=lambda f: {"color": f["properties"]["color"], "weight": f["properties"]["weight"], "opacity": 0.7},
                               tooltip=folium.GeoJsonTooltip(["label", "n"], aliases=["v/v_free", "số cạnh"])).add_to(fmap)
                fmap.get_root().add_child(legend_element())
            if o is not None and dst is not None:
                la, lo = [g.node_lat[o], g.node_lat[dst]], [g.node_lon[o], g.node_lon[dst]]
                fmap.fit_bounds([[min(la) - 0.004, min(lo) - 0.004], [max(la) + 0.004, max(lo) + 0.004]])
                for edges in alts:
                    folium.PolyLine([tuple(p) for e in edges for p in g.edge_coords[e]], color="#9467bd", weight=3,
                                    opacity=0.6, dash_array="6 6", tooltip="Tuyến thay thế").add_to(fmap)
                rl_col = {lab: RL_COLORS[i % len(RL_COLORS)] for i, lab in enumerate(rl_opts)}
                for lab, edges in routes.items():
                    base_lab = lab.split(" — ")[0]
                    pts = [tuple(p) for e in edges for p in g.edge_coords[e]]
                    if pts:
                        folium.PolyLine(pts, color=CLASSIC.get(base_lab, rl_col.get(base_lab, "#000")),
                                        weight=6 if lab.startswith("Dijkstra") else 4, opacity=0.85, tooltip=lab).add_to(fmap)
            if o is not None:
                folium.Marker([g.node_lat[o], g.node_lon[o]], tooltip="Điểm đi", icon=folium.Icon(color="green")).add_to(fmap)
            if dst is not None:
                folium.Marker([g.node_lat[dst], g.node_lon[dst]], tooltip="Điểm đến", icon=folium.Icon(color="red")).add_to(fmap)
            for inc in ss.incidents:
                folium.Marker(g.edge_coords[inc["edge"]].mean(axis=0).tolist(), icon=folium.Icon(color="purple", icon="warning-sign"),
                              tooltip=f"Sự cố ×{inc['severity']:.2f}, {inc['duration_min']:.0f} phút").add_to(fmap)
            out = st_folium(fmap, height=650, use_container_width=True, returned_objects=["last_clicked"], key="map")
            click = (out or {}).get("last_clicked")
            if click and click != ss.last_click:
                ss.last_click = click
                lat, lon = click["lat"], click["lng"]
                if click_mode == "Đặt điểm đi":
                    ss.origin = g.nearest_node(lat, lon)
                elif click_mode == "Đặt điểm đến":
                    ss.dest = g.nearest_node(lat, lon)
                else:
                    ss.incidents.append(Incident(g.nearest_edge(lat, lon), t0 - 5, float(dur), float(sev)).to_dict())
                st.rerun()
        with col_info:
            if o is None or dst is None:
                st.info("Nhấp bản đồ để chọn điểm đi và điểm đến (hoặc nhập toạ độ ở thanh bên).")
            else:
                st.markdown(f"**Đi:** nút {o} — **Đến:** nút {dst} — chim bay {g.straight_dist(o, dst) / 1000:.2f} km — "
                            f"xuất phát {hour:02d}:{minute:02d}{' (giờ thực)' if live else ''}, "
                            f"{len(ss.incidents)} sự cố, σ = {sigma}")
                st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
                st.caption("Dự kiến = theo dự báo V_base (+ sự cố đã biết); Thực = mô phỏng trong V_real; "
                           "P10/P50/P90 = phân vị thời gian của tuyến trên ngân hàng kịch bản nhiễu.")
                t_start = time.perf_counter()
                hrs = list(range(24))
                fast_edges = [td_astar(g, o, dst, 60.0 * h, forecast).edges for h in hrs]
                df = pd.DataFrame({"giờ": hrs, "Nhanh nhất (TD-A*)": [forecast.path_travel_time(e, 60.0 * h) for h, e in zip(hrs, fast_edges)],
                                   "Ngắn nhất (Dijkstra)": [forecast.path_travel_time(sp.edges, 60.0 * h) for h in hrs]})
                long = df.melt("giờ", var_name="tuyến", value_name="phút")
                chart = alt.Chart(long).mark_line(point=True).encode(
                    x=alt.X("giờ:Q", title="Giờ xuất phát"), y=alt.Y("phút:Q", title="Thời gian đi (phút)"), color="tuyến:N")
                if V_bank is not None:
                    q10q90 = [np.quantile(path_times(V_bank, g.length, e, 60.0 * h), [0.1, 0.9]) for h, e in zip(hrs, fast_edges)]
                    bdf = pd.DataFrame({"giờ": hrs, "p10": [x[0] for x in q10q90], "p90": [x[1] for x in q10q90]})
                    chart = alt.Chart(bdf).mark_area(opacity=0.25, color="#1f77b4").encode(
                        x="giờ:Q", y=alt.Y("p10:Q", title="Thời gian đi (phút)"), y2="p90:Q") + chart
                chart = chart + alt.Chart(pd.DataFrame({"giờ": [t0 / 60.0]})).mark_rule(color="red", strokeDash=[4, 3]).encode(x="giờ:Q")
                st.markdown("**Thời gian đi của cặp O–D theo 24 giờ xuất phát** (dải: P10–P90 của tuyến nhanh nhất)")
                st.altair_chart(chart, width="stretch")
                st.caption(f"Tính 24 lần TD-A* + dải tin cậy trong {1000 * (time.perf_counter() - t_start):.0f} ms.")
    with tab_net:
        import pydeck as pdk

        st.markdown(f"**Mức ùn tắc toàn mạng lúc {hour:02d}:{minute:02d}**{' (giờ thực)' if live else ''} — "
                    "màu theo v / v_free: " + " · ".join(f"<span style='color:{c}'>■</span> {lab}" for _, c, lab in CLASSES),
                    unsafe_allow_html=True)
        st.caption(f"Toàn bộ {g.n_edges:,} cạnh được vẽ bằng WebGL (pydeck); di chuột lên một đoạn để xem tên đường và vận tốc.".replace(",", "."))
        # Không dùng width_units: bản deck.gl đi kèm Streamlit không nhận giá trị này (widthUnits = undefined)
        # khiến nét vẽ phình thành các mảng khổng lồ. Độ rộng tính theo mét, chặn dưới 1 px để luôn nhìn thấy.
        layer = pdk.Layer("PathLayer", pydeck_paths(mode, int(t0)), get_path="path", get_color="color", get_width="width",
                          width_min_pixels=1, pickable=True)
        view = pdk.ViewState(latitude=float(g.node_lat.mean()), longitude=float(g.node_lon.mean()), zoom=12.5)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style=None,
                                 tooltip={"text": "{name}\n{v} km/h — v/v_free {ratio}"}), height=650)


main()
