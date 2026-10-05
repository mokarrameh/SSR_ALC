from __future__ import annotations

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None
from pathlib import Path
from typing import Any, Dict, Optional


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Install: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML must be a dict at the top level.")
    return data


def merge_yaml_into_defaults(defaults: Dict[str, Any], yaml_data: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Merge order: defaults < yaml['global'] < yaml[section]."""
    merged = dict(defaults)
    if isinstance(yaml_data.get("global"), dict):
        merged = _deep_update(merged, yaml_data["global"])
    if isinstance(yaml_data.get(section), dict):
        merged = _deep_update(merged, yaml_data[section])
    return merged