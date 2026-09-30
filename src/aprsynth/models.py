from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelOutput(StrictModel):
    error_description: str | None = None
    buggy_code: str
    fixed_code: str

    @field_validator("buggy_code", "fixed_code")
    @classmethod
    def require_non_empty_code(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("code must not be empty")
        return value


class VLLMServeOptions(StrictModel):
    require_launch: bool = False
    port: int | None = Field(default=None, gt=0)
    tensor_parallel_size: int = Field(default=1, gt=0)
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    max_model_len: int | None = Field(default=None, gt=0)
    cuda_visible_devices: str | None = None
    attention_backend: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    readiness_timeout: float = Field(default=600.0, gt=0)
    log_file: str | None = None


class ModelConfig(StrictModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = "http://localhost:8000/v1"
    name: str
    api_key: str | None = None
    request_timeout: float = Field(default=120.0, gt=0)
    run_name: str | None = None
    output_dir: str | None = None
    random_seed: int | None = None
    serve: VLLMServeOptions = Field(default_factory=VLLMServeOptions)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("name")
    @classmethod
    def require_model_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("model name must not be empty")
        return value


class GenerationOptions(StrictModel):
    num_examples: int = Field(default=100, gt=0)
    workers: int = Field(default=5, gt=0)
    temperature: float = Field(default=0.7, ge=0)
    max_tokens: int = Field(default=2048, gt=0)
    random_seed: int = 42
    sampling: Literal["stratified", "uniform"] = "stratified"
    languages: list[str] | None = None
    bugs: list[str] | None = None


class ValidationOptions(StrictModel):
    # retries after the first attempt
    max_retries: int = Field(default=2, ge=0)
    min_code_lines: int = Field(default=5, ge=0)
    require_difference: bool = True
    max_changed_line_ratio: float | None = Field(default=None, gt=0, le=1)


class EvaluationOptions(StrictModel):
    # None enables evaluation when evaluators are configured
    enabled: bool | None = None
    temperature: float = Field(default=0.2, ge=0)
    max_output_tokens: int = Field(default=512, gt=0)
    workers: int = Field(default=1, gt=0)
    max_retries: int = Field(default=2, ge=0)
    random_seed: int = 42


class SelectionOptions(StrictModel):
    threshold: float = Field(default=8.5, ge=0, le=10)
    weight_config: Literal[
        "balanced",
        "security_focused",
        "correctness_focused",
        "quality_focused",
        "equal_weights",
    ] = "balanced"


class GenerationConfig(StrictModel):
    run_name: str = "demo"
    output_dir: str = "synthetic_runs/demo"
    models: list[ModelConfig] = Field(min_length=1)
    evaluators: list[ModelConfig] = Field(default_factory=list)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    validation: ValidationOptions = Field(default_factory=ValidationOptions)
    evaluation: EvaluationOptions = Field(default_factory=EvaluationOptions)
    selection: SelectionOptions = Field(default_factory=SelectionOptions)

    @field_validator("run_name")
    @classmethod
    def require_run_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("run_name must not be empty")
        return value

    @model_validator(mode="after")
    def resolve_evaluation_enabled(self) -> GenerationConfig:
        has_evaluators = len(self.evaluators) > 0

        if self.evaluation.enabled is None:
            self.evaluation.enabled = has_evaluators
            return self

        if self.evaluation.enabled and not has_evaluators:
            raise ValueError("evaluation.enabled requires at least one evaluator")

        return self


class ValidationResult(StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class GenerationMetadata(StrictModel):
    model: str
    temperature: float
    seed: int | None = None
    attempt: int
    created_at: str
    language: str
    bug: str


class GeneratedSample(StrictModel):
    sample_id: str
    run_id: str
    prompt_version: str
    resolved_description: str
    generated_output: ModelOutput
    metadata: GenerationMetadata
    validation: ValidationResult


class InvalidSample(StrictModel):
    run_id: str
    prompt_version: str
    resolved_description: str
    metadata: GenerationMetadata
    validation: ValidationResult
    raw_output: str | None = None


class CodeEvaluation(StrictModel):
    correctness: int = Field(
        ge=0,
        le=10,
        description="Rating for correctness from 0 to 10",
    )
    code_quality: int = Field(
        ge=0,
        le=10,
        description="Rating for code quality from 0 to 10",
    )
    security: int = Field(
        ge=0,
        le=10,
        description="Rating for security from 0 to 10",
    )
    performance: int = Field(
        ge=0,
        le=10,
        description="Rating for performance from 0 to 10",
    )
    completeness: int = Field(
        ge=0,
        le=10,
        description="Rating for completeness from 0 to 10",
    )


class EvaluationResult(StrictModel):
    correctness: int | None = Field(default=None, ge=0, le=10)
    code_quality: int | None = Field(default=None, ge=0, le=10)
    security: int | None = Field(default=None, ge=0, le=10)
    performance: int | None = Field(default=None, ge=0, le=10)
    completeness: int | None = Field(default=None, ge=0, le=10)
    length: float = Field(default=0, ge=0)
    valid_format: bool
    error: str | None = None


class EvaluationMetadata(StrictModel):
    attempt: int
    created_at: str
    seed: int | None = None


class EvaluationRecord(StrictModel):
    sample_id: str
    run_id: str
    generated_by: str
    evaluator_model: str
    self_evaluation: bool
    prompt_version: str
    source_path: str
    error_description: str
    diff: str
    evaluation: EvaluationResult
    raw_output: str | None = None
    metadata: EvaluationMetadata


class ScoredSample(StrictModel):
    sample_id: str
    generated_by: str
    bug: str
    buggy_code: str
    fixed_code: str
    language: str
    diff: str
    error_description: str
    final_score: float
    selected: bool
