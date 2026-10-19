"""Tiện ích dùng chung: logging, seed, hình học trên mặt cầu, chuẩn hoá chuỗi, JSON."""
from __future__ import annotations

import json
import logging
import math
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Cấu hình logging ra console (UTF-8) và (tuỳ chọn) file."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
        except (AttributeError, ValueError):
            pass
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=_LOG_FORMAT, handlers=handlers, force=True)
    for noisy in ("matplotlib", "PIL", "urllib3", "fiona", "pyogrio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_global_seed(seed: int) -> None:
    """Cố định seed cho random, numpy và torch (nếu có)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def derive_seed(base: int, *keys: int | str) -> int:
    """Sinh seed con tất định từ seed gốc và các khoá (chuỗi hoặc số)."""
    ints = [base] + [k if isinstance(k, int) else int.from_bytes(str(k).encode()[:8].ljust(8, b"\0"), "little")
                     for k in keys]
    return int(np.random.SeedSequence(ints).generate_state(1)[0])


def haversine_m(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Khoảng cách chim bay (m) — hỗ trợ scalar hoặc numpy array."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Phiên bản thuần Python (nhanh hơn numpy cho từng cặp điểm)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def local_xy_m(lat: Any, lon: Any, lat0: float, lon0: float) -> tuple[Any, Any]:
    """Chiếu phẳng xấp xỉ (equirectangular) quanh (lat0, lon0): trả về (x đông, y bắc) theo mét."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return x, y


def normalize_text(s: str | None) -> str:
    """Bỏ dấu tiếng Việt, chữ thường, gộp khoảng trắng (dùng để khớp tên đường)."""
    if not s:
        return ""
    s = s.replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.lower().split())


def save_json(obj: Any, path: Path) -> None:
    """Ghi JSON UTF-8 (tự chuyển kiểu numpy)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Không tuần tự hoá được {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: Any, figures_dir: Path, name: str, dpi: int = 300) -> list[Path]:
    """Lưu hình matplotlib dạng PNG (dpi) và PDF."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = figures_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    return out
