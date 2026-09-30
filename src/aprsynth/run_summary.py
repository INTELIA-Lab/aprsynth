from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from . import __version__
from .config import write_resolved_config
from .evaluator import EvaluationSummary
from .generator import GenerationSummary
from .languages_and_bugs import get_bug_types, get_languages
from .models import GenerationConfig
from .prompts import EVALUATION_PROMPT_VERSION, PROMPT_VERSION
from .scoring import SelectionSummary
from .utils import utc_now, utc_stamp
from .writers import write_json

STAGES = ("generation", "evaluation", "selection")
STAGE_FILES = {
    "generation": ("samples", "invalid_samples"),
    "evaluation": ("evaluations",),
    "selection": ("selected_samples", "rejected_samples"),
}


def start_run(config: GenerationConfig) -> dict:
    now = utc_now()
    run_id = f"{config.run_name}-{utc_stamp()}"
    run_summary = {
        "run_id": run_id,
        "run_name": config.run_name,
        "created_at": now,
        "updated_at": now,
    }
    for stage in STAGES:
        run_summary[stage] = None

    run_summary["files"] = {
        "run_summary": str(run_summary_path(config)),
        "config": str(config_path(config)),
    }
    return run_summary


def resume_run(config: GenerationConfig) -> dict:
    path = run_summary_path(config)
    if not path.exists():
        return start_run(config)

    run_summary_text = path.read_text(encoding="utf-8")
    run_summary = json.loads(run_summary_text)
    if "run_id" not in run_summary or "generation" not in run_summary:
        return start_run(config)
    return run_summary


@contextmanager
def run_stage(
    config: GenerationConfig,
    run_summary: dict,
    stage: str,
) -> Iterator[None]:
    # a rerun stage makes the results of every later stage stale
    stage_index = STAGES.index(stage)
    later_stages = STAGES[stage_index + 1:]
    for later_stage in later_stages:
        run_summary[later_stage] = None
        for file_key in STAGE_FILES[later_stage]:
            run_summary["files"].pop(file_key, None)

    run_header = _run_header(config)
    run_summary.update(run_header)
    # completed_at stays empty if the stage stops before the end
    run_summary[stage] = {"started_at": utc_now(), "completed_at": None}
    write_resolved_config(config, config_path(config))
    write_run_summary(config, run_summary)

    yield

    run_summary[stage]["completed_at"] = utc_now()
    write_run_summary(config, run_summary)


def record_generation(
    config: GenerationConfig,
    run_summary: dict,
    summaries: list[GenerationSummary],
) -> None:
    runs = []
    for summary in summaries:
        runs.append(
            {
                "model": summary.model,
                "output_dir": str(summary.output_dir),
                "valid_samples": summary.valid_samples,
                "invalid_samples": summary.invalid_samples,
                "started_at": summary.started_at,
                "completed_at": summary.completed_at,
            }
        )

    requested_samples = config.generation.num_examples * len(config.models)
    valid_samples = sum(summary.valid_samples for summary in summaries)
    invalid_samples = sum(summary.invalid_samples for summary in summaries)
    run_summary["generation"].update(
        {
            "requested_samples": requested_samples,
            "valid_samples": valid_samples,
            "invalid_samples": invalid_samples,
            "runs": runs,
        }
    )

    samples_paths = [str(summary.samples_path) for summary in summaries]
    invalid_samples_paths = [str(summary.invalid_samples_path) for summary in summaries]
    run_summary["files"]["samples"] = samples_paths
    run_summary["files"]["invalid_samples"] = invalid_samples_paths
    write_run_summary(config, run_summary)


def record_evaluation(
    config: GenerationConfig,
    run_summary: dict,
    summaries: list[EvaluationSummary],
) -> None:
    runs = []
    for summary in summaries:
        runs.append(
            {
                "evaluator_model": summary.evaluator_model,
                "total_evaluations": summary.total_evaluations,
                "valid_evaluations": summary.valid_evaluations,
                "invalid_evaluations": summary.invalid_evaluations,
                "started_at": summary.started_at,
                "completed_at": summary.completed_at,
            }
        )

    total_evaluations = sum(summary.total_evaluations for summary in summaries)
    valid_evaluations = sum(summary.valid_evaluations for summary in summaries)
    invalid_evaluations = sum(summary.invalid_evaluations for summary in summaries)
    run_summary["evaluation"].update(
        {
            "total_evaluations": total_evaluations,
            "valid_evaluations": valid_evaluations,
            "invalid_evaluations": invalid_evaluations,
            "runs": runs,
        }
    )

    # every evaluator appends to the same file, so the first summary holds its path
    if summaries:
        run_summary["files"]["evaluations"] = str(summaries[0].evaluations_path)
    write_run_summary(config, run_summary)


def record_selection(
    config: GenerationConfig,
    run_summary: dict,
    summary: SelectionSummary,
) -> None:
    run_summary["selection"].update(
        {
            "threshold": config.selection.threshold,
            "weight_config": config.selection.weight_config,
            "total_samples": summary.total_samples,
            "selected_samples": summary.selected_samples,
            "rejected_samples": summary.rejected_samples,
        }
    )

    run_summary["files"]["selected_samples"] = str(summary.selected_path)
    run_summary["files"]["rejected_samples"] = str(summary.rejected_path)
    write_run_summary(config, run_summary)


def write_run_summary(config: GenerationConfig, run_summary: dict) -> None:
    run_summary["updated_at"] = utc_now()
    write_json(run_summary_path(config), run_summary)


def run_summary_path(config: GenerationConfig) -> Path:
    return Path(config.output_dir) / "run_summary.json"


def config_path(config: GenerationConfig) -> Path:
    return Path(config.output_dir) / "config.resolved.yaml"


def _run_header(config: GenerationConfig) -> dict:
    model_names = [model.name for model in config.models]
    evaluator_names = [evaluator.name for evaluator in config.evaluators]
    languages = config.generation.languages or get_languages()
    bugs = config.generation.bugs or get_bug_types()

    return {
        "aprsynth_version": __version__,
        "output_dir": config.output_dir,
        "models": model_names,
        "evaluators": evaluator_names,
        "prompt_versions": {
            "generation": PROMPT_VERSION,
            "evaluation": EVALUATION_PROMPT_VERSION,
        },
        "sampling": config.generation.sampling,
        "languages": languages,
        "bugs": bugs,
    }

