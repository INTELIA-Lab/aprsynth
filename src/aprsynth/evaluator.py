from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .client import OpenAICompatibleClient
from .diff import get_diff, get_num_lines
from .generator import build_model_generation_configs
from .models import (
    CodeEvaluation,
    EvaluationMetadata,
    EvaluationRecord,
    EvaluationResult,
    GeneratedSample,
    GenerationConfig,
    ModelConfig,
)
from .progress import progress_bar, status
from .prompts import EVALUATION_PROMPT_VERSION, render_evaluation_messages
from .server import with_launch_base_url
from .utils import stable_seed, utc_now
from .writers import append_jsonl, read_jsonl, reset_jsonl


@dataclass(frozen=True)
class EvaluationInput:
    sample: GeneratedSample
    source_path: Path


@dataclass(frozen=True)
class EvaluationSummary:
    output_dir: Path
    evaluations_path: Path
    evaluator_model: str
    total_evaluations: int
    valid_evaluations: int
    invalid_evaluations: int
    started_at: str
    completed_at: str


def evaluate_dataset(
    config: GenerationConfig,
    evaluator: ModelConfig,
    client: OpenAICompatibleClient,
    *,
    append: bool = True,
) -> EvaluationSummary:
    inputs = load_evaluation_inputs(config)
    return evaluate_inputs(config, evaluator, client, inputs, append=append)


def evaluate_inputs(
    config: GenerationConfig,
    evaluator: ModelConfig,
    client: OpenAICompatibleClient,
    inputs: list[EvaluationInput],
    *,
    append: bool = True,
) -> EvaluationSummary:
    started = utc_now()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations_path = output_dir / "evaluations.jsonl"
    if not append:
        reset_jsonl(evaluations_path)

    valid_count = 0
    invalid_count = 0
    workers = config.evaluation.workers
    evaluator_label = evaluator.run_name or evaluator.name
    progress_label = f"evaluate {evaluator_label}"
    status(
        f"evaluation started: {evaluator_label} "
        f"({len(inputs)} samples, {workers} workers)"
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for index, item in enumerate(inputs):
            future = executor.submit(evaluate_one, item, evaluator, config, client)
            future_to_index[future] = index

        indexed_records: list[tuple[int, EvaluationRecord]] = []
        for future in progress_bar(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=progress_label,
            unit="sample",
        ):
            index = future_to_index[future]
            indexed_records.append((index, future.result()))

    # write in input order so the output does not depend on thread timing
    indexed_records.sort(key=lambda pair: pair[0])
    for _, record in indexed_records:
        append_jsonl(evaluations_path, record)
        if record.evaluation.valid_format:
            valid_count += 1
        else:
            invalid_count += 1

    status(
        f"evaluation finished: {evaluator_label} "
        f"({valid_count} valid, {invalid_count} invalid)"
    )

    return EvaluationSummary(
        output_dir=output_dir,
        evaluations_path=evaluations_path,
        evaluator_model=evaluator.name,
        total_evaluations=len(inputs),
        valid_evaluations=valid_count,
        invalid_evaluations=invalid_count,
        started_at=started,
        completed_at=utc_now(),
    )


def evaluate_one(
    item: EvaluationInput,
    evaluator: ModelConfig,
    config: GenerationConfig,
    client: OpenAICompatibleClient,
) -> EvaluationRecord:
    output = item.sample.generated_output
    diff = get_diff(output.buggy_code, output.fixed_code)
    messages = render_evaluation_messages(
        buggy_code=output.buggy_code,
        patch=diff,
        error_description=item.sample.resolved_description,
    )
    buggy_lines = get_num_lines(output.buggy_code)
    fixed_lines = get_num_lines(output.fixed_code)
    code_length = buggy_lines + fixed_lines

    schema = CodeEvaluation.model_json_schema()
    last_raw_output: str | None = None
    last_error: str | None = None
    max_attempts = config.evaluation.max_retries + 1

    # requests that exceed the evaluator context fail and are kept as invalid
    for attempt in range(1, max_attempts + 1):
        seed = _evaluation_seed(config, item, evaluator, attempt)
        try:
            last_raw_output = client.chat(
                messages,
                temperature=config.evaluation.temperature,
                max_tokens=config.evaluation.max_output_tokens,
                seed=seed,
                guided_json=schema,
            )
            payload = json.loads(last_raw_output)
            parsed = CodeEvaluation.model_validate(payload)
            result = EvaluationResult(
                correctness=parsed.correctness,
                code_quality=parsed.code_quality,
                security=parsed.security,
                performance=parsed.performance,
                completeness=parsed.completeness,
                length=code_length,
                valid_format=True,
            )
            return _record(
                item,
                evaluator,
                result,
                diff=diff,
                raw_output=last_raw_output,
                seed=seed,
                attempt=attempt,
            )
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON: {exc}"
        except ValidationError as exc:
            last_error = f"schema validation failed: {exc}"
        except Exception as exc:
            last_error = f"evaluation failed: {exc}"

    result = EvaluationResult(
        length=code_length,
        valid_format=False,
        error=last_error or "evaluation failed",
    )
    last_seed = _evaluation_seed(config, item, evaluator, max_attempts)
    return _record(
        item,
        evaluator,
        result,
        diff=diff,
        raw_output=last_raw_output,
        seed=last_seed,
        attempt=max_attempts,
    )


def load_evaluation_inputs(config: GenerationConfig) -> list[EvaluationInput]:
    inputs: list[EvaluationInput] = []
    for path in generation_sample_paths(config):
        if not path.exists():
            continue

        for row in read_jsonl(path):
            sample = GeneratedSample.model_validate(row)
            inputs.append(EvaluationInput(sample=sample, source_path=path))

    return inputs


def generation_sample_paths(config: GenerationConfig) -> list[Path]:
    generation_configs = build_model_generation_configs(config)
    return [
        Path(generation_config.output_dir) / "samples.jsonl"
        for generation_config in generation_configs
    ]


def build_evaluator_configs(config: GenerationConfig) -> list[ModelConfig]:
    return [with_launch_base_url(evaluator) for evaluator in config.evaluators]


def _record(
    item: EvaluationInput,
    evaluator: ModelConfig,
    result: EvaluationResult,
    *,
    diff: str,
    raw_output: str | None,
    seed: int,
    attempt: int,
) -> EvaluationRecord:
    sample = item.sample
    generated_by = sample.metadata.model
    metadata = EvaluationMetadata(
        attempt=attempt,
        created_at=utc_now(),
        seed=seed,
    )
    return EvaluationRecord(
        sample_id=sample.sample_id,
        run_id=sample.run_id,
        generated_by=generated_by,
        evaluator_model=evaluator.name,
        self_evaluation=generated_by == evaluator.name,
        prompt_version=EVALUATION_PROMPT_VERSION,
        source_path=str(item.source_path),
        error_description=sample.resolved_description,
        diff=diff,
        evaluation=result,
        raw_output=raw_output,
        metadata=metadata,
    )


def _evaluation_seed(
    config: GenerationConfig,
    item: EvaluationInput,
    evaluator: ModelConfig,
    attempt: int,
) -> int:
    return stable_seed(
        config.evaluation.random_seed,
        item.sample.sample_id,
        item.sample.metadata.model,
        evaluator.name,
        attempt,
    )

