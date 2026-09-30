from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .evaluator import generation_sample_paths
from .models import EvaluationRecord, GenerationConfig, ScoredSample
from .writers import append_jsonl, read_jsonl, reset_jsonl

WEIGHT_CONFIGS = {
    "balanced": {
        "correctness": 0.3,
        "code_quality": 0.2,
        "security": 0.1,
        "performance": 0.1,
        "completeness": 0.1,
        "length": 0.2,
    },
    "security_focused": {
        "correctness": 0.2,
        "code_quality": 0.15,
        "security": 0.4,
        "performance": 0.1,
        "completeness": 0.1,
        "length": 0.05,
    },
    "correctness_focused": {
        "correctness": 0.5,
        "code_quality": 0.2,
        "security": 0.1,
        "performance": 0.05,
        "completeness": 0.1,
        "length": 0.05,
    },
    "quality_focused": {
        "correctness": 0.25,
        "code_quality": 0.4,
        "security": 0.1,
        "performance": 0.15,
        "completeness": 0.05,
        "length": 0.05,
    },
    "equal_weights": {
        "correctness": 1 / 6,
        "code_quality": 1 / 6,
        "security": 1 / 6,
        "performance": 1 / 6,
        "completeness": 1 / 6,
        "length": 1 / 6,
    },
}


@dataclass(frozen=True)
class SelectionSummary:
    output_dir: Path
    selected_path: Path
    rejected_path: Path
    total_samples: int
    selected_samples: int
    rejected_samples: int


def select_dataset(config: GenerationConfig) -> SelectionSummary:
    output_dir = Path(config.output_dir)
    evaluations_path = output_dir / "evaluations.jsonl"
    selected_path = output_dir / "selected_samples.jsonl"
    rejected_path = output_dir / "rejected_samples.jsonl"

    samples = _load_samples(config)
    evaluation_rows = read_jsonl(evaluations_path)
    records = [EvaluationRecord.model_validate(row) for row in evaluation_rows]
    scored = score_samples(
        samples=samples,
        records=records,
        threshold=config.selection.threshold,
        weight_config=config.selection.weight_config,
    )

    reset_jsonl(selected_path)
    reset_jsonl(rejected_path)
    selected_count = 0
    rejected_count = 0
    for sample in scored:
        exported = _export_sample(sample)
        if sample.selected:
            append_jsonl(selected_path, exported)
            selected_count += 1
        else:
            append_jsonl(rejected_path, exported)
            rejected_count += 1

    return SelectionSummary(
        output_dir=output_dir,
        selected_path=selected_path,
        rejected_path=rejected_path,
        total_samples=len(scored),
        selected_samples=selected_count,
        rejected_samples=rejected_count,
    )


def score_samples(
    *,
    samples: dict[str, dict],
    records: list[EvaluationRecord],
    threshold: float,
    weight_config: str,
) -> list[ScoredSample]:
    """Score and deduplicate samples exactly like the companion study."""
    weights = WEIGHT_CONFIGS[weight_config]
    records = _drop_repeated_evaluations(records, samples)
    lengths = [record.evaluation.length for record in records]
    max_length = max(lengths, default=0)

    groups: dict[tuple[str, ...], dict] = {}
    for record in records:
        sample = samples.get(record.sample_id)
        if sample is None:
            continue

        flat_sample = _flatten_sample(sample, record)
        pair_key = _sample_key(flat_sample)
        group_key = (
            *pair_key,
            flat_sample["language"],
            flat_sample["generated_by"],
        )
        if group_key not in groups:
            groups[group_key] = {"sample": flat_sample, "scores": []}
        group = groups[group_key]

        if not _has_scores(record):
            continue
        score = _weighted_score(record, weights, max_length)
        group["scores"].append(score)

    kept_groups = _one_group_per_pair(groups)

    scored_samples: list[ScoredSample] = []
    for group in kept_groups:
        sample = group["sample"]
        final_score = _mean(group["scores"])
        scored_sample = ScoredSample(
            sample_id=sample["sample_id"],
            generated_by=sample["generated_by"],
            bug=sample["bug"],
            buggy_code=sample["buggy_code"],
            fixed_code=sample["fixed_code"],
            language=sample["language"],
            diff=sample["diff"],
            error_description=sample["error_description"],
            final_score=round(final_score, 4),
            selected=final_score >= threshold,
        )
        scored_samples.append(scored_sample)

    scored_samples.sort(key=lambda item: item.final_score, reverse=True)
    return scored_samples


def _drop_repeated_evaluations(
    records: list[EvaluationRecord],
    samples: dict[str, dict],
) -> list[EvaluationRecord]:
    # one score per evaluator, generator and pair, as in the study
    seen: set[tuple[str, ...]] = set()
    unique_records: list[EvaluationRecord] = []
    for record in records:
        sample = samples.get(record.sample_id)
        if sample is None:
            continue

        flat_sample = _flatten_sample(sample, record)
        pair_key = _sample_key(flat_sample)
        key = (record.generated_by, record.evaluator_model, *pair_key)
        if key in seen:
            continue

        seen.add(key)
        unique_records.append(record)

    return unique_records


def _one_group_per_pair(groups: dict[tuple[str, ...], dict]) -> list[dict]:
    # sorted keys make the generator that sorts first win each pair
    kept: dict[tuple[str, str, str], dict] = {}
    for group_key in sorted(groups):
        group = groups[group_key]
        pair_key = _sample_key(group["sample"])
        if pair_key in kept:
            continue
        kept[pair_key] = group

    return list(kept.values())


def _load_samples(config: GenerationConfig) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    for path in generation_sample_paths(config):
        if not path.exists():
            continue

        for row in read_jsonl(path):
            samples[row["sample_id"]] = row

    return samples


def _flatten_sample(sample: dict, record: EvaluationRecord) -> dict:
    generated_output = sample["generated_output"]
    metadata = sample["metadata"]
    buggy_code = generated_output["buggy_code"]
    fixed_code = generated_output["fixed_code"]
    return {
        "sample_id": sample["sample_id"],
        "generated_by": record.generated_by,
        "bug": metadata["bug"],
        "buggy_code": buggy_code,
        "fixed_code": fixed_code,
        "language": metadata["language"],
        "diff": record.diff,
        "error_description": sample["resolved_description"],
    }


def _sample_key(sample: dict) -> tuple[str, str, str]:
    return sample["bug"], sample["buggy_code"], sample["fixed_code"]


def _weighted_score(
    record: EvaluationRecord,
    weights: dict[str, float],
    max_length: float,
) -> float:
    evaluation = record.evaluation
    length_score = _length_score(evaluation.length, max_length)
    values = {
        "correctness": evaluation.correctness or 0,
        "code_quality": evaluation.code_quality or 0,
        "security": evaluation.security or 0,
        "performance": evaluation.performance or 0,
        "completeness": evaluation.completeness or 0,
        "length": length_score,
    }
    return sum(weights[name] * values[name] for name in weights)


def _has_scores(record: EvaluationRecord) -> bool:
    evaluation = record.evaluation
    criteria_scores = (
        evaluation.correctness,
        evaluation.code_quality,
        evaluation.security,
        evaluation.performance,
        evaluation.completeness,
    )
    return all(score is not None for score in criteria_scores)


def _length_score(length: float, max_length: float) -> float:
    if max_length <= 0:
        return 0.0
    return math.log1p(length) / math.log1p(max_length) * 10


def _export_sample(sample: ScoredSample) -> dict:
    return sample.model_dump(mode="json", exclude={"selected"})


def _mean(values) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)
