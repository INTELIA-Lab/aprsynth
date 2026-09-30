from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def stable_seed(base_seed: int, *parts: object) -> int:
    seed_parts = [str(part) for part in (base_seed, *parts)]
    seed_text = "|".join(seed_parts)
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def utc_now() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0).isoformat()


def utc_stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")
