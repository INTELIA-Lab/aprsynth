from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator

from tqdm.auto import tqdm


def progress_bar(
    items: Iterable,
    *,
    total: int,
    desc: str,
    unit: str,
) -> Iterator:
    # hide the bar when stderr is redirected to a file or a pipe
    yield from tqdm(
        items,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )


def status(message: str) -> None:
    # tqdm.write keeps the message from breaking an active progress bar
    tqdm.write(f"[aprsynth] {message}", file=sys.stderr)
