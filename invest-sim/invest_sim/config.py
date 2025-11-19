from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .data_models import BacktestConfig, SimulationConfig


def load_config(path: str | Path) -> SimulationConfig:
    """从 JSON/YAML 文件加载前瞻性模拟配置。"""

    path = Path(path)
    raw = _load_raw_config(path)
    try:
        return SimulationConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"配置文件验证失败：{exc}") from exc


def load_backtest_config(path: str | Path) -> BacktestConfig:
    """从 JSON/YAML 文件加载历史回测配置。"""

    path = Path(path)
    raw = _load_raw_config(path)
    try:
        return BacktestConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"配置文件验证失败：{exc}") from exc


def _load_raw_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix in {".json"}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要额外安装 pyyaml 才能读取 YAML 配置") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    raise ValueError(f"暂不支持的配置格式：{suffix}")

