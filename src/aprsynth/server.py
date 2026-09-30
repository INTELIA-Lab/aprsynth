from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse, urlunparse

import requests

from .models import ModelConfig

SERVER_HOST = "0.0.0.0"
READINESS_INTERVAL_SECONDS = 2
SHUTDOWN_TIMEOUT_SECONDS = 30
# gives the GPUs time to free their memory before the next model starts
COOLDOWN_SECONDS = 30


@dataclass(frozen=True)
class VLLMServerConfig:
    model: str
    port: int = 8000
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    api_key: str | None = None
    cuda_visible_devices: str | None = None
    attention_backend: str | None = None
    extra_args: list[str] = field(default_factory=list)
    log_file: str | None = None


@dataclass
class RunningVLLMServer:
    process: subprocess.Popen
    log_handle: TextIO | None = None

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_handle is not None:
            self.log_handle.close()
        time.sleep(COOLDOWN_SECONDS)


def build_vllm_command(config: VLLMServerConfig) -> list[str]:
    command = [
        "vllm",
        "serve",
        config.model,
        "--host",
        SERVER_HOST,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--dtype",
        config.dtype,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]

    if config.max_model_len:
        command.extend(["--max-model-len", str(config.max_model_len)])
    if config.api_key:
        command.extend(["--api-key", config.api_key])

    command.extend(config.extra_args)
    return command


def build_vllm_environment(config: VLLMServerConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # finds vllm in the same environment even when it is not activated
    interpreter_bin_dir = str(Path(sys.executable).parent)
    current_path = env.get("PATH", "")
    env["PATH"] = f"{interpreter_bin_dir}{os.pathsep}{current_path}"

    if config.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    if config.attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = config.attention_backend

    return env


def wait_until_ready(
    base_url: str,
    *,
    timeout: float,
    process: subprocess.Popen,
) -> bool:
    deadline = time.monotonic() + timeout
    models_url = f"{base_url.rstrip('/')}/models"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            response = requests.get(models_url, timeout=READINESS_INTERVAL_SECONDS)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(READINESS_INTERVAL_SECONDS)

    return False


def server_config_from_model(model_config: ModelConfig) -> VLLMServerConfig:
    serve = model_config.serve
    port = serve.port or port_from_base_url(model_config.base_url) or 8000
    return VLLMServerConfig(
        model=model_config.name,
        port=port,
        tensor_parallel_size=serve.tensor_parallel_size,
        dtype=serve.dtype,
        gpu_memory_utilization=serve.gpu_memory_utilization,
        max_model_len=serve.max_model_len,
        api_key=model_config.api_key,
        cuda_visible_devices=serve.cuda_visible_devices,
        attention_backend=serve.attention_backend,
        extra_args=serve.extra_args,
        log_file=serve.log_file,
    )


def port_from_base_url(base_url: str) -> int | None:
    parsed = urlparse(base_url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


def with_launch_base_url(model: ModelConfig) -> ModelConfig:
    if not model.serve.require_launch or model.serve.port is None:
        return model

    base_url = _base_url_with_port(model.base_url, model.serve.port)
    return model.model_copy(update={"base_url": base_url})


def start_vllm_process(config: VLLMServerConfig) -> RunningVLLMServer:
    env = build_vllm_environment(config)
    vllm_executable = shutil.which("vllm", path=env["PATH"])
    if vllm_executable is None:
        raise RuntimeError(
            "vllm executable not found. Install vLLM in this environment or point "
            "the config at an endpoint that is already running."
        )

    command = build_vllm_command(config)
    command[0] = vllm_executable
    log_handle = _open_log_file(config.log_file)
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_handle,
        stderr=log_handle,
    )
    return RunningVLLMServer(process=process, log_handle=log_handle)


def _base_url_with_port(base_url: str, port: int) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    # IPv6 literals need brackets inside a netloc
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    netloc = f"{host}:{port}"
    path = parsed.path or "/v1"
    url = urlunparse((parsed.scheme, netloc, path, "", "", ""))
    return url.rstrip("/")


def _open_log_file(path: str | None) -> TextIO | None:
    if path is None:
        return None
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("w", encoding="utf-8")
