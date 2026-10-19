"""Giai đoạn 1 — tải, làm sạch và chuẩn hoá mạng đường lái xe quận Hà Đông từ OpenStreetMap.

Đầu ra (data/processed):
    graph.npz       cấu trúc CSR + thuộc tính số (numpy)
    nodes.parquet   bảng nút
    edges.parquet   bảng cạnh (edge_idx, u, v, length, highway_group, v_free, name, ref, geometry WKT, ...)
và results/graph_stats.json, reports/figures/graph_map.{png,pdf}.

Chạy: python -m src.graph_builder [--mode smoke|full] [--force-download]
"""
from __future__ import annotations

import argparse
import ast
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.utils import haversine_m, save_figure, save_json, setup_logging

log = logging.getLogger("graph_builder")

GROUP_ORDER: list[str] = ["trunk", "primary", "secondary", "tertiary", "residential"]


class NetworkUnavailableError(RuntimeError):
    """Không tải được dữ liệu OSM và không có bản cache."""


# ---------------------------------------------------------------------------
# Ranh giới khu vực
# ---------------------------------------------------------------------------
def _project_area_km2(gdf: Any) -> float:
    import osmnx as ox

    return float(ox.projection.project_gdf(gdf).area.sum() / 1e6)


def get_boundary(cfg: Config) -> tuple[Any, str]:
    """Trả về (polygon WGS84 đã buffer, mô tả nguồn).

    Thứ tự: truy vấn tên quận -> hợp các phường mới -> boundary đã cache -> bounding box.
    """
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import box
    from shapely.ops import unary_union

    gcfg = cfg["graph"]
    raw_dir = cfg.path("raw_dir")
    cache = raw_dir / gcfg["boundary_name"]
    min_area = float(gcfg["min_boundary_area_km2"])
    ox.settings.requests_timeout = int(gcfg["requests_timeout_s"])

    poly_gdf, source = None, ""
    for q in gcfg["place_queries"]:
        try:
            g = ox.geocode_to_gdf(q)
            area = _project_area_km2(g)
            if g.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon") and area >= min_area:
                poly_gdf, source = g[["geometry"]], f"geocode:{q} ({area:.1f} km2)"
                break
            log.warning("Geocode '%s' trả về vùng %.1f km2 < %.1f km2 — bỏ qua", q, area, min_area)
        except Exception as exc:  # noqa: BLE001 - mọi lỗi mạng/geocode đều fallback
            log.warning("Geocode '%s' thất bại: %s", q, exc)

    if poly_gdf is None:
        parts = []
        for q in gcfg["fallback_place_union"]:
            try:
                parts.append(ox.geocode_to_gdf(q).geometry.iloc[0])
            except Exception as exc:  # noqa: BLE001
                log.warning("Geocode phường '%s' thất bại: %s", q, exc)
        if parts:
            g = gpd.GeoDataFrame(geometry=[unary_union(parts)], crs="EPSG:4326")
            area = _project_area_km2(g)
            if area >= min_area:
                poly_gdf, source = g, f"union_of_{len(parts)}_wards ({area:.1f} km2)"

    if poly_gdf is None and cache.exists():
        poly_gdf = gpd.read_file(cache)
        source = f"cached_boundary:{cache.name}"
        log.warning("Dùng ranh giới đã cache %s", cache)
        return poly_gdf.geometry.iloc[0], source

    if poly_gdf is None:
        w, s, e, n = gcfg["fallback_bbox"]
        poly_gdf = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326")
        source = f"fallback_bbox:{gcfg['fallback_bbox']}"
        log.warning("Dùng bounding box dự phòng %s", gcfg["fallback_bbox"])

    projected = ox.projection.project_gdf(poly_gdf)
    buffered = projected.buffer(float(gcfg["buffer_m"])).to_crs("EPSG:4326")
    out = gpd.GeoDataFrame(geometry=[buffered.iloc[0]], crs="EPSG:4326")
    out["source"] = source
    out["core_wkt"] = poly_gdf.geometry.iloc[0].wkt
    out.to_file(cache, driver="GeoJSON")
    return out.geometry.iloc[0], source


def download_or_load_graph(cfg: Config, force: bool = False) -> tuple[nx.MultiDiGraph, str]:
    """Tải graph từ OSM (hoặc đọc graphml đã cache)."""
    import osmnx as ox

    gcfg = cfg["graph"]
    gpath = cfg.path("raw_dir") / gcfg["graphml_name"]
    if gpath.exists() and not force:
        log.info("Đọc graph đã cache: %s", gpath)
        return ox.load_graphml(gpath), f"cache:{gpath.name}"
    try:
        poly, source = get_boundary(cfg)
        log.info("Tải mạng '%s' trong vùng (%s, buffer %s m)...", gcfg["network_type"], source, gcfg["buffer_m"])
        G, errors = None, []
        for url in gcfg.get("overpass_urls") or [ox.settings.overpass_url]:
            ox.settings.overpass_url = url
            try:
                G = ox.graph_from_polygon(poly, network_type=gcfg["network_type"], simplify=True,
                                          retain_all=False, truncate_by_edge=True)
                source += f" | overpass:{url}"
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("Overpass %s lỗi: %s", url, exc)
                errors.append(f"{url}: {exc}")
        if G is None:
            raise RuntimeError("; ".join(errors))
    except Exception as exc:  # noqa: BLE001
        if gpath.exists():
            log.error("Tải OSM thất bại (%s) — dùng bản graphml đã cache %s", exc, gpath)
            return ox.load_graphml(gpath), f"cache_after_error:{gpath.name}"
        raise NetworkUnavailableError(
            f"Không tải được dữ liệu OSM ({exc}). Hãy kiểm tra kết nối mạng hoặc đặt sẵn file "
            f"{gpath} (graphml) rồi chạy lại.") from exc
    ox.save_graphml(G, gpath)
    log.info("Đã cache graph vào %s", gpath)
    return G, source


# ---------------------------------------------------------------------------
# Chuẩn hoá thuộc tính
# ---------------------------------------------------------------------------
def _as_list(val: Any) -> list[Any]:
    """Thuộc tính OSM có thể là scalar, list, hoặc chuỗi biểu diễn list (sau khi đọc graphml)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, (list, tuple, set)):
        return list(val)
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        except (ValueError, SyntaxError):
            return [val]
    return [val]


def highway_group(highway: Any, groups: dict[str, list[str]]) -> str:
    """Gom tag `highway` (có thể là list) về nhóm có thứ hạng cao nhất."""
    lookup = {h: g for g, hs in groups.items() for h in hs}
    found = [lookup.get(str(h), "residential") for h in _as_list(highway)] or ["residential"]
    return min(found, key=GROUP_ORDER.index)


_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_maxspeed(val: Any) -> float | None:
    """Parse maxspeed: '50', '50 mph', ['60','50'], '40;50' -> km/h (lấy giá trị nhỏ nhất). None nếu không parse được."""
    speeds: list[float] = []
    for item in _as_list(val):
        s = str(item).strip().lower()
        for tok in re.split(r"[;|,]", s):
            m = _NUM_RE.search(tok)
            if not m:
                continue
            v = float(m.group(1))
            if "mph" in tok:
                v *= 1.609344
            speeds.append(v)
    return min(speeds) if speeds else None


def _join_names(val: Any) -> str:
    return " / ".join(dict.fromkeys(str(x) for x in _as_list(val) if str(x).strip()))


def clean_graph(G: nx.MultiDiGraph, cfg: Config) -> tuple[nx.DiGraph, dict[str, int]]:
    """Bỏ self-loop, xử lý cạnh song song, giữ thành phần liên thông mạnh lớn nhất."""
    report: dict[str, int] = {"raw_nodes": G.number_of_nodes(), "raw_edges": G.number_of_edges()}
    G = G.copy()
    loops = list(nx.selfloop_edges(G, keys=True))
    G.remove_edges_from(loops)
    report["self_loops_removed"] = len(loops)

    policy = cfg["graph"]["parallel_edge_policy"]
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes(data=True))
    n_parallel = 0
    for u, v, data in G.edges(data=True):
        if D.has_edge(u, v):
            n_parallel += 1
            old = D[u][v]
            if float(data.get("length", np.inf)) < float(old.get("length", np.inf)):
                if policy == "merge":
                    data = dict(data)
                    data["name"] = _join_names(_as_list(old.get("name")) + _as_list(data.get("name")))
                D[u][v].clear()
                D[u][v].update(data)
            elif policy == "merge":
                old["name"] = _join_names(_as_list(old.get("name")) + _as_list(data.get("name")))
        else:
            D.add_edge(u, v, **data)
    report["parallel_edges_collapsed"] = n_parallel

    scc = max(nx.strongly_connected_components(D), key=len)
    D = D.subgraph(scc).copy()
    report["scc_nodes"] = D.number_of_nodes()
    report["scc_edges"] = D.number_of_edges()
    report["nodes_dropped_not_in_scc"] = report["raw_nodes"] - D.number_of_nodes()

    assert nx.is_strongly_connected(D), "Đồ thị sau làm sạch phải liên thông mạnh"
    assert nx.number_of_selfloops(D) == 0
    return D, report


def build_tables(D: nx.DiGraph, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Gán chỉ số liên tục và tạo bảng nút/cạnh đã chuẩn hoá."""
    from shapely.geometry import LineString

    gcfg = cfg["graph"]
    defaults = gcfg["default_maxspeed_kmh"]
    lo, hi = gcfg["maxspeed_bounds_kmh"]
    ff = float(gcfg["free_flow_factor"])

    osmids = sorted(D.nodes())
    idx_of = {n: i for i, n in enumerate(osmids)}
    nodes = pd.DataFrame({
        "node_idx": np.arange(len(osmids)),
        "osmid": osmids,
        "lat": [float(D.nodes[n]["y"]) for n in osmids],
        "lon": [float(D.nodes[n]["x"]) for n in osmids],
    })
    lat = nodes["lat"].to_numpy()
    lon = nodes["lon"].to_numpy()

    rows = []
    n_tagged = 0
    for u, v, d in D.edges(data=True):
        ui, vi = idx_of[u], idx_of[v]
        group = highway_group(d.get("highway"), gcfg["highway_groups"])
        ms = parse_maxspeed(d.get("maxspeed"))
        tagged = ms is not None
        if tagged:
            n_tagged += 1
            ms = float(np.clip(ms, lo, hi))
        else:
            ms = float(defaults[group])
        geom = d.get("geometry")
        if geom is None:
            geom = LineString([(lon[ui], lat[ui]), (lon[vi], lat[vi])])
        oneway = d.get("oneway", False)
        if isinstance(oneway, str):
            oneway = oneway.lower() in ("true", "yes", "1")
        rows.append({
            "u": ui, "v": vi, "length": float(d.get("length", 0.0)),
            "highway_group": group, "maxspeed_kmh": ms, "maxspeed_tagged": tagged,
            "v_free": ms * ff, "name": _join_names(d.get("name")), "ref": _join_names(d.get("ref")),
            "oneway": bool(oneway), "geometry_wkt": geom.wkt,
        })
    edges = pd.DataFrame(rows).sort_values(["u", "v"], kind="stable").reset_index(drop=True)
    edges.insert(0, "edge_idx", np.arange(len(edges)))
    # chiều dài phải >= khoảng cách chim bay giữa hai đầu (để heuristic admissible)
    straight = haversine_m(lat[edges.u], lon[edges.u], lat[edges.v], lon[edges.v])
    edges["length"] = np.maximum(edges["length"].to_numpy(), straight)
    edges.loc[edges["length"] <= 0.0, "length"] = 0.1
    edges["u_lat"], edges["u_lon"] = lat[edges.u], lon[edges.u]
    edges["v_lat"], edges["v_lon"] = lat[edges.v], lon[edges.v]
    edges["mid_lat"] = (edges.u_lat + edges.v_lat) / 2
    edges["mid_lon"] = (edges.u_lon + edges.v_lon) / 2
    dy = np.radians(edges.v_lat - edges.u_lat)
    dx = np.radians(edges.v_lon - edges.u_lon) * np.cos(np.radians(edges.mid_lat))
    edges["bearing_deg"] = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    info = {"maxspeed_tag_coverage": n_tagged / max(len(edges), 1)}
    return nodes, edges, info


def build_csr(n_nodes: int, edges: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cấu trúc CSR: các cạnh ra của nút u là out_edges[out_ptr[u]:out_ptr[u+1]]."""
    u = edges["u"].to_numpy(np.int64)
    order = np.argsort(u, kind="stable")
    counts = np.bincount(u, minlength=n_nodes)
    out_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=out_ptr[1:])
    return {
        "out_ptr": out_ptr,
        "out_edges": edges["edge_idx"].to_numpy(np.int64)[order],
        "edge_from": u,
        "edge_to": edges["v"].to_numpy(np.int64),
    }


def save_processed(cfg: Config, nodes: pd.DataFrame, edges: pd.DataFrame, csr: dict[str, np.ndarray]) -> None:
    out = cfg.path("processed_dir")
    nodes.to_parquet(out / "nodes.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)
    group_id = edges["highway_group"].map({g: i for i, g in enumerate(GROUP_ORDER)}).to_numpy(np.int8)
    np.savez_compressed(
        out / "graph.npz", **csr,
        length=edges["length"].to_numpy(np.float64), v_free=edges["v_free"].to_numpy(np.float64),
        group_id=group_id, node_lat=nodes["lat"].to_numpy(), node_lon=nodes["lon"].to_numpy(),
        node_osmid=nodes["osmid"].to_numpy(np.int64), bearing=edges["bearing_deg"].to_numpy(),
        group_names=np.array(GROUP_ORDER),
    )


def compute_stats(nodes: pd.DataFrame, edges: pd.DataFrame, csr: dict[str, np.ndarray],
                  clean_report: dict[str, int], info: dict[str, Any], source: str) -> dict[str, Any]:
    deg = np.diff(csr["out_ptr"])
    by_group = edges.groupby("highway_group").agg(n_edges=("edge_idx", "size"), length_km=("length", lambda s: s.sum() / 1000))
    by_group = by_group.reindex(GROUP_ORDER).fillna(0)
    return {
        "source": source, **clean_report,
        "n_nodes": int(len(nodes)), "n_edges": int(len(edges)),
        "total_length_km": float(edges["length"].sum() / 1000),
        "mean_edge_length_m": float(edges["length"].mean()),
        "max_out_degree": int(deg.max()), "mean_out_degree": float(deg.mean()),
        "maxspeed_tag_coverage": float(info["maxspeed_tag_coverage"]),
        "bbox": [float(nodes.lon.min()), float(nodes.lat.min()), float(nodes.lon.max()), float(nodes.lat.max())],
        "by_group": {g: {"n_edges": int(r.n_edges), "length_km": float(r.length_km)} for g, r in by_group.iterrows()},
        "named_edges_share": float((edges["name"] != "").mean()),
    }


GROUP_COLORS = {"trunk": "#d7191c", "primary": "#fdae61", "secondary": "#2b83ba",
                "tertiary": "#7fbc41", "residential": "#bdbdbd"}
GROUP_WIDTH = {"trunk": 2.2, "primary": 1.8, "secondary": 1.4, "tertiary": 1.0, "residential": 0.5}


def edge_line_segments(edges: pd.DataFrame) -> list[np.ndarray]:
    """Danh sách mảng toạ độ (lon, lat) của từng cạnh — dùng cho LineCollection."""
    from shapely import wkt

    return [np.asarray(wkt.loads(w).coords) for w in edges["geometry_wkt"]]


def plot_graph(cfg: Config, edges: pd.DataFrame, stats: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    segs = edge_line_segments(edges)
    fig, ax = plt.subplots(figsize=(8, 8))
    for g in reversed(GROUP_ORDER):
        mask = (edges["highway_group"] == g).to_numpy()
        lc = LineCollection([s for s, m in zip(segs, mask) if m], colors=GROUP_COLORS[g], linewidths=GROUP_WIDTH[g])
        ax.add_collection(lc)
    bpath = cfg.path("raw_dir") / cfg["graph"]["boundary_name"]
    if bpath.exists():
        import geopandas as gpd
        from shapely import wkt as _wkt

        b = gpd.read_file(bpath)
        if "core_wkt" in b.columns:
            gpd.GeoSeries([_wkt.loads(b["core_wkt"].iloc[0])], crs="EPSG:4326").boundary.plot(
                ax=ax, color="k", linewidth=1.0, linestyle="--")
    ax.autoscale()
    ax.set_aspect(1 / math.cos(math.radians(edges["mid_lat"].mean())))
    ax.set_xlabel("Kinh độ")
    ax.set_ylabel("Vĩ độ")
    ax.set_title(f"Mạng đường lái xe Hà Đông (+buffer): {stats['n_nodes']} nút, {stats['n_edges']} cạnh, "
                 f"{stats['total_length_km']:.0f} km", fontsize=10)
    handles = [Line2D([0], [0], color=GROUP_COLORS[g], lw=GROUP_WIDTH[g] + 1, label=g) for g in GROUP_ORDER]
    handles.append(Line2D([0], [0], color="k", lw=1, ls="--", label="ranh giới quận (cũ)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save_figure(fig, cfg.path("figures_dir"), "graph_map", dpi=cfg["evaluation"]["figure_dpi"])
    plt.close(fig)


def run(cfg: Config, force_download: bool = False) -> dict[str, Any]:
    """Chạy toàn bộ giai đoạn 1."""
    G, source = download_or_load_graph(cfg, force=force_download)
    D, clean_report = clean_graph(G, cfg)
    nodes, edges, info = build_tables(D, cfg)
    csr = build_csr(len(nodes), edges)
    save_processed(cfg, nodes, edges, csr)
    stats = compute_stats(nodes, edges, csr, clean_report, info, source)
    save_json(stats, cfg.path("results_dir") / "graph_stats.json")
    plot_graph(cfg, edges, stats)
    log.info("Đồ thị: %d nút, %d cạnh, %.1f km; max out-degree %d; maxspeed có tag %.1f%%",
             stats["n_nodes"], stats["n_edges"], stats["total_length_km"], stats["max_out_degree"],
             100 * stats["maxspeed_tag_coverage"])
    for g, r in stats["by_group"].items():
        log.info("  %-12s %5d cạnh  %7.1f km", g, r["n_edges"], r["length_km"])
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.mode)
    setup_logging(cfg["project"]["log_level"])
    run(cfg, force_download=args.force_download)


if __name__ == "__main__":
    main()
