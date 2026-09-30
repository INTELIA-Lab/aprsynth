from __future__ import annotations

import json
from typing import Any

import requests

from .models import ModelConfig


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def check_ready(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=min(self.config.request_timeout, 10),
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        guided_json: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "temperature": temperature,
            "n": 1,
            "stream": False,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if guided_json is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": guided_json},
            }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.config.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            response_preview = json.dumps(data)[:500]
            raise RuntimeError(
                f"Unexpected API response: {response_preview}"
            ) from exc

        content = message.get("content")
        if content is not None:
            return content

        finish_reason = choice.get("finish_reason")
        reasoning = message.get("reasoning_content")
        if finish_reason == "length" and reasoning:
            raise RuntimeError(
                f"Model returned empty content: response truncated at "
                f"max_tokens={max_tokens} while still emitting "
                f"reasoning_content ({len(reasoning)} chars). Increase "
                f"max_tokens or disable thinking for this model."
            )
        response_preview = json.dumps(data)[:500]
        raise RuntimeError(
            f"Model returned no content (finish_reason={finish_reason}): "
            f"{response_preview}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

