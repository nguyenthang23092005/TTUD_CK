"""Nạp cấu hình YAML, gộp ghi đè theo chế độ (smoke/full) và phân giải đường dẫn."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: Path = PROJECT_ROOT / "configs" / "config.yaml"
VALID_MODES = ("smoke", "full")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gộp đệ quy `override` vào bản sao của `base` (dict lồng nhau)."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


class Config(dict):
    """dict có thêm tiện ích: `cfg.path("results_dir")` trả về Path tuyệt đối đã tạo thư mục."""

    @property
    def mode(self) -> str:
        return self["_mode"]

    def path(self, key: str, create: bool = True) -> Path:
        p = PROJECT_ROOT / self["paths"][key]
        if create and not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(mode: str = "full", path: Path | str | None = None) -> Config:
    """Đọc config.yaml và áp ghi đè của `modes.<mode>`.

    Args:
        mode: "smoke" hoặc "full".
        path: đường dẫn file cấu hình (mặc định configs/config.yaml).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode phải thuộc {VALID_MODES}, nhận '{mode}'")
    # HADONG_CONFIG cho phép trỏ tới file cấu hình khác (thí nghiệm) mà không sửa mã
    cfg_path = Path(path) if path else Path(os.environ.get("HADONG_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    modes = raw.pop("modes", {}) or {}
    merged = deep_merge(raw, modes.get(mode, {}) or {})
    merged["_mode"] = mode
    return Config(merged)
