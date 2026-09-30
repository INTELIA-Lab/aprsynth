from __future__ import annotations

import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .client import OpenAICompatibleClient
from .languages_and_bugs import get_bug_types, get_languages
from .models import (
    GeneratedSample,
    GenerationConfig,
    GenerationMetadata,
    InvalidSample,
    ModelConfig,
    ModelOutput,
    ValidationResult,
)
from .progress import progress_bar, status
from .prompts import PROMPT_VERSION, render_generation_messages
from .server import with_launch_base_url
from .utils import stable_seed, utc_now, utc_stamp
from .validators import validate_model_output
from .writers import append_jsonl, reset_jsonl

TAGGED_OUTPUT_PATTERN = re.compile(
    r"<error_description>\s*(.*?)\s*</error_description>\s*"
    r"<buggy_code>\s*(.*?)\s*</buggy_code>\s*"
    r"<fixed_code>\s*(.*?)\s*</fixed_code>",
    re.DOTALL,
)


@dataclass(frozen=True)
class GenerationJob:
    index: int
    language: str
    bug: str


@dataclass(frozen=True)
class GenerationSummary:
    run_id: str
    model: str
    output_dir: Path
    samples_path: Path
    invalid_samples_path: Path
    valid_samples: int
    invalid_samples: int
    started_at: str
    completed_at: str


def generate_dataset(
    config: GenerationConfig,
    client: OpenAICompatibleClient,
    *,
    run_id: str | None = None,
) -> GenerationSummary:
    started = utc_now()
    model_config = require_active_model(config)
    if run_id is None:
        run_id = f"{config.run_name}-{utc_stamp()}"
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.jsonl"
    invalid_samples_path = output_dir / "invalid_samples.jsonl"
    reset_jsonl(samples_path)
    reset_jsonl(invalid_samples_path)

    jobs = build_jobs(config)
    valid_count = 0
    invalid_count = 0
    workers = config.generation.workers
    model_label = model_config.run_name or model_config.name
    progress_desc = f"generate {model_label}"
    status(
        f"generation started: {model_label} "
        f"({len(jobs)} samples, {workers} workers)"
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {}
        for job in jobs:
            future = executor.submit(generate_one, job, config, client, run_id)
            future_to_job[future] = job

        results: list[tuple[int, GeneratedSample | InvalidSample]] = []
        completed_futures = as_completed(future_to_job)
        futures_in_progress = progress_bar(
            completed_futures,
            total=len(future_to_job),
            desc=progress_desc,
            unit="sample",
        )
        for future in futures_in_progress:
            job = future_to_job[future]
            results.append((job.index, future.result()))

    # write in job order so the output does not depend on thread timing
    ordered_results = sorted(results, key=lambda item: item[0])
    for _, result in ordered_results:
        if isinstance(result, GeneratedSample):
            append_jsonl(samples_path, result)
            valid_count += 1
        else:
            append_jsonl(invalid_samples_path, result)
            invalid_count += 1

    status(
        f"generation finished: {model_label} "
        f"({valid_count} valid, {invalid_count} invalid)"
    )

    return GenerationSummary(
        run_id=run_id,
        model=model_config.name,
        output_dir=output_dir,
        samples_path=samples_path,
        invalid_samples_path=invalid_samples_path,
        valid_samples=valid_count,
        invalid_samples=invalid_count,
        started_at=started,
        completed_at=utc_now(),
    )


def build_jobs(
    config: GenerationConfig,
) -> list[GenerationJob]:
    rng = random.Random(config.generation.random_seed)
    languages = config.generation.languages or get_languages()
    bugs = config.generation.bugs or get_bug_types()
    if not languages:
        raise ValueError("at least one language is required")
    if not bugs:
        raise ValueError("at least one bug type is required")

    num_examples = config.generation.num_examples
    if config.generation.sampling == "uniform":
        pairs = _uniform_pairs(rng, languages, bugs, num_examples)
    else:
        pairs = _stratified_pairs(rng, languages, bugs, num_examples)

    jobs: list[GenerationJob] = []
    for index, (language, bug) in enumerate(pairs):
        jobs.append(GenerationJob(index=index, language=language, bug=bug))
    return jobs


def _uniform_pairs(
    rng: random.Random,
    languages: list[str],
    bugs: list[str],
    num_examples: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for _ in range(num_examples):
        language = rng.choice(languages)
        bug = rng.choice(bugs)
        pairs.append((language, bug))
    return pairs


def _stratified_pairs(
    rng: random.Random,
    languages: list[str],
    bugs: list[str],
    num_examples: int,
) -> list[tuple[str, str]]:
    grid: list[tuple[str, str]] = []
    for language in languages:
        for bug in bugs:
            grid.append((language, bug))

    full_rounds, remainder = divmod(num_examples, len(grid))
    pairs = grid * full_rounds
    pairs.extend(_balanced_partial_round(rng, languages, bugs, remainder))

    # shuffle so the request order does not follow the grid
    rng.shuffle(pairs)
    return pairs


def _balanced_partial_round(
    rng: random.Random,
    languages: list[str],
    bugs: list[str],
    count: int,
) -> list[tuple[str, str]]:
    shuffled_languages = list(languages)
    shuffled_bugs = list(bugs)
    rng.shuffle(shuffled_languages)
    rng.shuffle(shuffled_bugs)

    # consecutive bugs from a circular list keep both marginals within one
    base_share, extra = divmod(count, len(shuffled_languages))
    chosen: list[tuple[str, str]] = []
    bug_position = 0
    for language_index, language in enumerate(shuffled_languages):
        share = base_share
        if language_index < extra:
            share += 1
        for _ in range(share):
            bug = shuffled_bugs[bug_position % len(shuffled_bugs)]
            chosen.append((language, bug))
            bug_position += 1
    return chosen


def build_model_generation_configs(
    config: GenerationConfig,
) -> list[GenerationConfig]:
    generation_configs: list[GenerationConfig] = []
    has_multiple_models = len(config.models) > 1

    for configured_model in config.models:
        model = with_launch_base_url(configured_model)
        model_label = model.run_name or _model_slug(model.name)

        if model.run_name:
            run_name = model.run_name
        elif has_multiple_models:
            run_name = f"{config.run_name}-{model_label}"
        else:
            run_name = config.run_name

        if model.output_dir:
            output_dir = model.output_dir
        elif has_multiple_models:
            output_dir = str(Path(config.output_dir) / model_label)
        else:
            output_dir = config.output_dir

        generation = config.generation
        if model.random_seed is not None:
            seed_update = {"random_seed": model.random_seed}
            generation = generation.model_copy(update=seed_update)

        model_update = {
            "models": [model],
            "run_name": run_name,
            "output_dir": output_dir,
            "generation": generation,
        }
        model_generation_config = config.model_copy(update=model_update)
        generation_configs.append(model_generation_config)

    return generation_configs


def require_active_model(config: GenerationConfig) -> ModelConfig:
    if len(config.models) != 1:
        raise ValueError("generation requires an effective single-model config")
    return config.models[0]


def generate_one(
    job: GenerationJob,
    config: GenerationConfig,
    client: OpenAICompatibleClient,
    run_id: str,
) -> GeneratedSample | InvalidSample:
    messages = render_generation_messages(job.language, job.bug)
    last_raw_output: str | None = None
    last_errors: list[str] = []
    max_attempts = config.validation.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        seed = _generation_seed(config, job, attempt)
        metadata = _metadata(config, job, attempt, seed)
        try:
            last_raw_output = client.chat(
                messages,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_tokens,
                seed=seed,
            )
            model_output = parse_tagged_generation_output(last_raw_output)
            validation = validate_model_output(model_output, config.validation)
            if validation.valid:
                job_description = _job_description(job)
                sample_id = compute_sample_id(job_description, model_output)
                resolved_description = model_output.error_description or job_description
                return GeneratedSample(
                    sample_id=sample_id,
                    run_id=run_id,
                    prompt_version=PROMPT_VERSION,
                    resolved_description=resolved_description,
                    generated_output=model_output,
                    metadata=metadata,
                    validation=validation,
                )
            last_errors = validation.errors
        except ValueError as exc:
            last_errors = [str(exc)]
        except ValidationError as exc:
            last_errors = [f"structure validation failed: {exc}"]
        except Exception as exc:
            last_errors = [f"generation failed: {exc}"]

    job_description = _job_description(job)
    final_seed = _generation_seed(config, job, max_attempts)
    final_metadata = _metadata(config, job, max_attempts, final_seed)
    failed_validation = ValidationResult(valid=False, errors=last_errors)
    return InvalidSample(
        run_id=run_id,
        prompt_version=PROMPT_VERSION,
        resolved_description=job_description,
        metadata=final_metadata,
        validation=failed_validation,
        raw_output=last_raw_output,
    )


def parse_tagged_generation_output(output_text: str) -> ModelOutput:
    # reasoning models may mention the tags first, so parse from the last opening tag
    last_opening_tag = output_text.rfind("<error_description>")
    start = max(0, last_opening_tag)
    match = TAGGED_OUTPUT_PATTERN.search(output_text, start)
    if not match:
        raise ValueError("output does not match the required tag format")

    error_description = match.group(1).strip()
    buggy_code = _strip_code_fence(match.group(2))
    fixed_code = _strip_code_fence(match.group(3))

    return ModelOutput(
        error_description=error_description,
        buggy_code=buggy_code,
        fixed_code=fixed_code,
    )


def _strip_code_fence(code: str) -> str:
    # some models wrap the code in markdown fences despite the prompt
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def compute_sample_id(
    resolved_description: str,
    output: ModelOutput,
) -> str:
    payload = {
        "resolved_description": resolved_description,
        "generated_output": output.model_dump(mode="json"),
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    return digest[:24]


def _metadata(
    config: GenerationConfig,
    job: GenerationJob,
    attempt: int,
    seed: int,
) -> GenerationMetadata:
    model_config = require_active_model(config)
    return GenerationMetadata(
        model=model_config.name,
        temperature=config.generation.temperature,
        seed=seed,
        attempt=attempt,
        created_at=utc_now(),
        language=job.language,
        bug=job.bug,
    )


def _generation_seed(
    config: GenerationConfig,
    job: GenerationJob,
    attempt: int,
) -> int:
    model_config = require_active_model(config)
    return stable_seed(
        config.generation.random_seed,
        model_config.name,
        job.index,
        job.language,
        job.bug,
        attempt,
    )


def _job_description(job: GenerationJob) -> str:
    return f"{job.bug} in {job.language}"


def _model_slug(model_name: str) -> str:
    dashed_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name)
    slug = dashed_name.strip("-_.")
    return slug.lower() or "model"

