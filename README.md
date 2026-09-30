# APRSynth

APRSynth is an open-source Python tool that uses large language models (LLMs) to build training data for automated program repair. From a single configuration file, it asks one or more generator models for pairs of buggy and fixed code, balanced across 12 programming languages and 13 bug categories. An ensemble of LLM evaluators scores each pair on five quality criteria, and only the pairs whose weighted score reaches a configurable threshold are kept.

Every request is sent with a fixed seed and each run saves its configuration and a summary of every stage, so a run can be repeated with identical requests and each sample traced back to its origin. APRSynth packages the pipeline of our previous study ([de-Fitero-Dominguez et al., 2026](https://doi.org/10.1016/j.eswa.2026.132154)).

## Pipeline

The pipeline has three stages that communicate only through files in the run directory.

1. **Generation.** Each request targets a language and a bug category and asks the model for an error description, a buggy snippet, and a fixed snippet. Outputs that break the validation rules are retried with a new seed.
2. **Evaluation.** Each evaluator rates the fix from 0 to 10 on correctness, code quality, security, performance, and completeness.
3. **Selection.** A weighted score combines the five criteria with a length score, and the samples that reach the threshold form the final dataset.

APRSynth works with any endpoint compatible with the OpenAI API and can also serve the models locally with vLLM, one at a time, so ensembles of several large models fit on a single GPU node.

## Installation

APRSynth requires Python 3.10 or later and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/INTELIA-Lab/aprsynth.git
cd aprsynth
uv sync # remote endpoints only
uv sync --extra serve # also serve models locally with vLLM
```

## Usage

The demo configuration generates 100 examples with a single locally served model, evaluates them with the same model, and keeps the samples whose score reaches 8.5.

```bash
uv run aprsynth pipeline --config configs/demo.yaml
```

Stages can also be run one at a time, for instance to filter the same pool again with another threshold or weight preset.

```bash
uv run aprsynth generate --config configs/demo.yaml
uv run aprsynth evaluate --config configs/demo.yaml
uv run aprsynth select --config configs/demo.yaml
```

The repository includes two configurations.

| Configuration | Purpose |
| --- | --- |
| `configs/demo.yaml` | Complete run with one locally served model. |
| `configs/paper.yaml` | Setup of our previous study, with six models and 5,000 samples per model. |

A configuration sets the models (`models`, `evaluators`), the languages and bug categories (`generation.languages`, `generation.bugs`), the validation rules (`validation`), and the selection policy (`selection.threshold`, `selection.weight_config`). When a model block sets `serve.require_launch: true`, vLLM is launched for that model and stopped afterwards. Set it to `false` to use an endpoint that is already running. The default languages and bug categories are listed in `src/aprsynth/languages_and_bugs.py`.

## Outputs

Each run writes its results to `synthetic_runs/<run_name>/`, including the selected and rejected samples, all the evaluations, a copy of the configuration as used, and a run summary. Every sample keeps the same identifier across these files, so any selected pair can be traced back to its evaluations and its generation request.

## Limitations

APRSynth only checks the form of the generated code and never compiles or runs it, so it establishes that a sample is plausible, not that it is semantically correct. Quality assessment relies on LLM judges, so the final dataset is bounded by the chosen generator and evaluator models.

## Citation

If you use APRSynth in your research, please cite our previous study:

```bibtex
@article{deFitero2026selfbootstrapping,
  author  = {de-Fitero-Dominguez, David and Garcia-Cabot, Antonio and Garcia-Lopez, Eva},
  title   = {Self-Bootstrapping Automated Program Repair: Using LLMs to Generate and Evaluate Synthetic Training Data for Bug Repair},
  journal = {Expert Systems with Applications},
  volume  = {319},
  pages   = {132154},
  year    = {2026},
  doi     = {10.1016/j.eswa.2026.132154}
}
```

## License

APRSynth is licensed under the Apache License 2.0. See `LICENSE.txt` for details.
