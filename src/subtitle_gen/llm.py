from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import LLMConfig, env_llm_model


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    content: str
    parsed: Any


class OpenAICompatibleLLM:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def complete_json(self, system_prompt: str, payload: Any) -> LLMResponse:
        content = self.complete_text(system_prompt, json.dumps(payload, ensure_ascii=False))
        return LLMResponse(content=content, parsed=parse_json_content(content))

    async def complete_json_async(self, system_prompt: str, payload: Any) -> LLMResponse:
        return await asyncio.to_thread(self.complete_json, system_prompt, payload)

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        model = env_llm_model(self.config)
        if not model:
            raise LLMError(
                "LLM model is not configured. Set [llm].model or SUBTITLE_GEN_LLM_MODEL."
            )
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("Missing LLM API key. Set [llm].api_key or OPENAI_API_KEY.")

        body = {
            "model": model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return self._post_chat_completions(body, api_key)

    async def complete_text_async(self, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(self.complete_text, system_prompt, user_prompt)

    def _post_chat_completions(self, body: dict[str, Any], api_key: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        encoded = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        last_error_message: str | None = None
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                last_error = exc
                last_error_message = _format_http_error(exc)
                if exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                last_error_message = _format_exception(exc)
            if attempt < self.config.max_retries:
                time.sleep(min(2.0**attempt, 8.0))
        raise LLMError(f"LLM request failed: {last_error_message or last_error}") from last_error


def parse_json_content(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = min(
            [index for index in [stripped.find("["), stripped.find("{")] if index >= 0],
            default=-1,
        )
        end = max(stripped.rfind("]"), stripped.rfind("}"))
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise LLMError("LLM response is not valid JSON.")


def _strip_code_fence(content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return content
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _format_http_error(exc: urllib.error.HTTPError) -> str:
    message = f"HTTP {exc.code} {exc.reason}"
    body = _read_error_body(exc)
    if body:
        message = f"{message}: {body}"
    return message


def _read_error_body(exc: urllib.error.HTTPError, max_chars: int = 600) -> str:
    try:
        raw_body = exc.read()
    except Exception:
        return ""
    if not raw_body:
        return ""
    encoding = exc.headers.get_content_charset() or "utf-8"
    try:
        body = raw_body.decode(encoding, errors="replace")
    except LookupError:
        body = raw_body.decode("utf-8", errors="replace")
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > max_chars:
        return f"{body[:max_chars]}..."
    return body


def _format_exception(exc: Exception) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
