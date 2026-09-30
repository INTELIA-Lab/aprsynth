from __future__ import annotations

LANGUAGES = [
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "Ruby",
    "Go",
    "Pascal",
    "Swift",
    "Kotlin",
    "Rust",
    "PHP",
    "C#",
]

BUG_TYPES = [
    "Off-by-one errors",
    "Infinite loops",
    "Incorrect loop boundaries",
    "Resource leaks (e.g., open files or sockets not being closed)",
    "Concurrency issues (e.g., race conditions, deadlocks)",
    "Improper error handling (e.g., missing exceptions or uncaught exceptions)",
    "Incorrect conditionals (e.g., wrong boolean expressions)",
    "SQL injections or unsafe user input handling",
    "Memory leaks or buffer overflows (for C/C++ code)",
    "Misuse of data structures (e.g., incorrect initialization or access)",
    "Incorrect use of API methods or functions",
    (
        "Inconsistent state in object-oriented code "
        "(e.g., failing to update object attributes correctly)"
    ),
    "Arithmetic errors (e.g., division by zero or integer overflows)",
]


def get_languages() -> list[str]:
    return list(LANGUAGES)


def get_bug_types() -> list[str]:
    return list(BUG_TYPES)
