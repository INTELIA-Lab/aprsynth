from __future__ import annotations

import difflib

from .models import ModelOutput, ValidationOptions, ValidationResult


def validate_model_output(
    output: ModelOutput,
    options: ValidationOptions,
) -> ValidationResult:
    errors: list[str] = []
    buggy_code = output.buggy_code
    fixed_code = output.fixed_code
    min_lines = options.min_code_lines

    buggy_lines = _non_blank_line_count(buggy_code)
    if buggy_lines < min_lines:
        errors.append(f"buggy code has fewer than {min_lines} lines")

    fixed_lines = _non_blank_line_count(fixed_code)
    if fixed_lines < min_lines:
        errors.append(f"fixed code has fewer than {min_lines} lines")

    compact_buggy_code = _without_whitespace(buggy_code)
    compact_fixed_code = _without_whitespace(fixed_code)
    has_code_change = compact_buggy_code != compact_fixed_code
    if options.require_difference and not has_code_change:
        errors.append("buggy and fixed code differ only in whitespace")

    max_ratio = options.max_changed_line_ratio
    if max_ratio is not None:
        ratio = changed_line_ratio(buggy_code, fixed_code)
        if ratio > max_ratio:
            errors.append(
                f"fix rewrites {ratio:.0%} of the lines, "
                f"above the {max_ratio:.0%} limit"
            )

    return ValidationResult(valid=not errors, errors=errors)


def changed_line_ratio(buggy_code: str, fixed_code: str) -> float:
    """Fraction of lines that the fix does not keep from the buggy version."""
    buggy_lines = _stripped_non_blank_lines(buggy_code)
    fixed_lines = _stripped_non_blank_lines(fixed_code)
    longest_side = max(len(buggy_lines), len(fixed_lines))
    if longest_side == 0:
        return 0.0

    matcher = difflib.SequenceMatcher(a=buggy_lines, b=fixed_lines, autojunk=False)
    kept_lines = 0
    for block in matcher.get_matching_blocks():
        kept_lines += block.size
    return 1 - kept_lines / longest_side


def _non_blank_line_count(code: str) -> int:
    return len(_stripped_non_blank_lines(code))


def _stripped_non_blank_lines(code: str) -> list[str]:
    lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


def _without_whitespace(code: str) -> str:
    return "".join(code.split())
