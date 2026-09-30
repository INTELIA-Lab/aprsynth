from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .client import OpenAICompatibleClient
from .evaluator import EvaluationSummary, build_evaluator_configs, evaluate_dataset
from .generator import (
    GenerationSummary,
    build_model_generation_configs,
    generate_dataset,
    require_active_model,
)
from .models import GenerationConfig, ModelConfig
from .progress import status
from .run_summary import (
    record_evaluation,
    record_generation,
    record_selection,
    resume_run,
    run_stage,
    start_run,
)
from .scoring import SelectionSummary, select_dataset
from .server import server_config_from_model, start_vllm_process, wait_until_ready


def run_generation(config: GenerationConfig) -> list[GenerationSummary]:
    run_summary = start_run(config)
    with run_stage(config, run_summary, "generation"):
        summaries = generate_all(config, run_summary)
    return summaries


def run_evaluation(config: GenerationConfig) -> list[EvaluationSummary]:
    run_summary = resume_run(config)
    with run_stage(config, run_summary, "evaluation"):
        summaries = evaluate_all(config, run_summary)
    return summaries


def run_selection(config: GenerationConfig) -> SelectionSummary:
    run_summary = resume_run(config)
    with run_stage(config, run_summary, "selection"):
        summary = select_dataset(config)
        record_selection(config, run_summary, summary)
    print_selection_summary(summary)
    return summary


def run_pipeline(config: GenerationConfig) -> None:
    run_summary = start_run(config)
    with run_stage(config, run_summary, "generation"):
        generate_all(config, run_summary)

    if not config.evaluation.enabled:
        return

    with run_stage(config, run_summary, "evaluation"):
        evaluate_all(config, run_summary)

    with run_stage(config, run_summary, "selection"):
        summary = select_dataset(config)
        record_selection(config, run_summary, summary)
    print_selection_summary(summary)


def generate_all(
    config: GenerationConfig,
    run_summary: dict,
) -> list[GenerationSummary]:
    generation_configs = build_model_generation_configs(config)
    total_models = len(generation_configs)
    run_id = run_summary["run_id"]

    summaries: list[GenerationSummary] = []
    for index, generation_config in enumerate(generation_configs, start=1):
        model_config = require_active_model(generation_config)
        if total_models > 1:
            status(f"model {index}/{total_models}: {model_config.name}")

        summary = generate_for_model(generation_config, run_id=run_id)
        summaries.append(summary)
        record_generation(config, run_summary, summaries)

    for summary in summaries:
        print_generation_summary(summary)
    return summaries


def evaluate_all(
    config: GenerationConfig,
    run_summary: dict,
) -> list[EvaluationSummary]:
    evaluator_configs = build_evaluator_configs(config)
    if not evaluator_configs:
        raise SystemExit("config must define evaluators to run evaluation")

    total_evaluators = len(evaluator_configs)

    summaries: list[EvaluationSummary] = []
    for index, evaluator in enumerate(evaluator_configs, start=1):
        if total_evaluators > 1:
            status(f"evaluator {index}/{total_evaluators}: {evaluator.name}")

        # all evaluators share one evaluations file, so only the first one truncates it
        append_to_file = index > 1
        summary = evaluate_for_evaluator(config, evaluator, append=append_to_file)
        summaries.append(summary)
        record_evaluation(config, run_summary, summaries)

    for summary in summaries:
        print_evaluation_summary(summary)
    return summaries


def generate_for_model(config: GenerationConfig, *, run_id: str) -> GenerationSummary:
    model_config = require_active_model(config)
    with _served_client(model_config) as client:
        return generate_dataset(config, client, run_id=run_id)


def evaluate_for_evaluator(
    config: GenerationConfig,
    evaluator: ModelConfig,
    *,
    append: bool,
) -> EvaluationSummary:
    with _served_client(evaluator) as client:
        return evaluate_dataset(config, evaluator, client, append=append)


@contextmanager
def _served_client(model_config: ModelConfig) -> Iterator[OpenAICompatibleClient]:
    running_server = None
    try:
        if model_config.serve.require_launch:
            server_config = server_config_from_model(model_config)
            status(f"launching vLLM: {model_config.name}")
            running_server = start_vllm_process(server_config)
            server_ready = wait_until_ready(
                model_config.base_url,
                timeout=model_config.serve.readiness_timeout,
                process=running_server.process,
            )
            if not server_ready:
                raise SystemExit(
                    f"vLLM server did not become ready at {model_config.base_url}"
                )
            status(f"vLLM ready: {model_config.base_url}")

        client = OpenAICompatibleClient(model_config)
        if not client.check_ready():
            raise SystemExit(
                f"OpenAI-compatible server is not ready at {model_config.base_url}"
            )

        yield client
    finally:
        if running_server is not None:
            status(f"stopping vLLM: {model_config.name}")
            running_server.stop()


def print_generation_summary(summary: GenerationSummary) -> None:
    status(
        "generation summary: "
        f"valid={summary.valid_samples}, invalid={summary.invalid_samples}, "
        f"output={summary.output_dir}"
    )


def print_evaluation_summary(summary: EvaluationSummary) -> None:
    status(
        "evaluation summary: "
        f"total={summary.total_evaluations}, valid={summary.valid_evaluations}, "
        f"invalid={summary.invalid_evaluations}"
    )
    status(f"evaluations: {summary.evaluations_path}")


def print_selection_summary(summary: SelectionSummary) -> None:
    status(
        "selection summary: "
        f"selected={summary.selected_samples}, rejected={summary.rejected_samples}"
    )
    status(f"selected: {summary.selected_path}")
    status(f"rejected: {summary.rejected_path}")
