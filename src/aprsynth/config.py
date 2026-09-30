from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import GenerationConfig


def load_config(path: str | Path) -> GenerationConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = _read_mapping(config_path)
    return GenerationConfig.model_validate(data)


def write_resolved_config(config: GenerationConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        data = config.model_dump(mode="json")
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            allow_unicode=False,
        )


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Config files must be YAML (.yaml or .yml)")

    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object")

    return data
