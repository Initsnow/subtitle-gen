from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import LLMConfig
from .llm import OpenAICompatibleLLM, StreamCallback
from .types import SubtitleItem


PROOFREAD_SYSTEM_PROMPT = """You are a subtitle proofreading engine.
Fix ASR errors and transcription artifacts.
Return only valid JSON in compact format: [{"i":<id>,"t":"<corrected text>"}, ...].
Only include items you actually corrected; omit unchanged ones.
Do not change meaning."""

ProgressCallback = Callable[[str], None]


class ProofreadError(RuntimeError):
    pass


class SubtitleProofreader:
    def __init__(
        self,
        llm: OpenAICompatibleLLM,
        config: LLMConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or LLMConfig()
        self.progress = progress

    def proofread(
        self,
        items: list[SubtitleItem],
    ) -> list[SubtitleItem]:
        corrected: list[SubtitleItem] = []
        batches = _batched(items, self.config.batch_size)
        for index, batch in enumerate(batches, start=1):
            label = f"Proofread batch {index}/{len(batches)} ({len(batch)} items)"
            with _stream(self.progress, label) as stream:
                _report(self.progress, f"proofreading batch {index}/{len(batches)} ({len(batch)} subtitle(s))")
                corrections = self._proofread_batch(batch, on_token=stream.on_token)
            corrected.extend(
                item.with_proofread(corrections[item.id]) if item.id in corrections else item
                for item in batch
            )
        return corrected

    def _proofread_batch(
        self,
        items: list[SubtitleItem],
        on_token: StreamCallback | None = None,
    ) -> dict[int, str]:
        payload: dict[str, Any] = {
            "requirements": [
                "Fix ASR errors and transcription artifacts in the following subtitle lines.",
                "Only return items that actually need correction; omit unchanged ones.",
                "Do not merge, split, remove, or reorder ids.",
                "Do not change meaning or rewrite for style.",
            ],
            "items": [{"i": item.id, "t": item.text} for item in items],
        }
        response = self.llm.complete_json(PROOFREAD_SYSTEM_PROMPT, payload, on_token=on_token)
        corrections = _validate_proofread_output(response.parsed, [item.id for item in items])
        if corrections is None:
            raise ProofreadError("LLM proofread output failed id validation.")
        return corrections


def _validate_proofread_output(raw_output: Any, expected_ids: list[int]) -> dict[int, str] | None:
    if isinstance(raw_output, dict) and "items" in raw_output:
        raw_output = raw_output["items"]
    if not isinstance(raw_output, list):
        return None

    expected_set = set(expected_ids)
    corrections: dict[int, str] = {}
    for item in raw_output:
        if not isinstance(item, dict):
            return None
        item_id = item.get("i")
        text = item.get("t")
        if not isinstance(item_id, int) or not isinstance(text, str):
            return None
        if item_id in corrections or item_id not in expected_set:
            return None
        corrections[item_id] = text.strip()

    return corrections


def _batched(items: list[SubtitleItem], batch_size: int) -> list[list[SubtitleItem]]:
    if batch_size <= 0:
        batch_size = len(items) or 1
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


class _NoopStream:
    def __enter__(self) -> _NoopStream:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    @staticmethod
    def on_token(token: str) -> None:
        pass


def _stream(progress: ProgressCallback | None, label: str) -> _NoopStream:
    factory = getattr(progress, "stream_context", None)
    if factory is not None:
        return factory(label)  # type: ignore[return-value]
    return _NoopStream()


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
