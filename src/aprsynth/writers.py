from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json(path: str | Path, item: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_jsonable(item)
    text = json.dumps(data, indent=2, sort_keys=False)
    output_path.write_text(text + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, item: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        data = _to_jsonable(item)
        line = json.dumps(data, sort_keys=False)
        handle.write(line)
        handle.write("\n")


def reset_jsonl(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break

    return items


def _to_jsonable(item: Any) -> Any:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    return item
