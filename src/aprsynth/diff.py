from __future__ import annotations

import difflib


def get_diff(buggy_code: str, fixed_code: str, ctx: int = 3) -> str:
    diff = difflib.unified_diff(
        buggy_code.splitlines(),
        fixed_code.splitlines(),
        fromfile="buggy_code",
        tofile="fixed_code",
        lineterm="",
        n=ctx,
    )
    return "\n".join(diff)


def get_num_lines(code: str) -> int:
    return len(code.splitlines())
