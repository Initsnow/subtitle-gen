from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import LLMConfig
from .llm import OpenAICompatibleLLM
from .types import SubtitleItem


TRANSLATION_SYSTEM_PROMPT = """You are a subtitle translation engine.
Return only valid JSON.
Preserve ids exactly and translate each subtitle line concisely."""

ProgressCallback = Callable[[str], None]


class TranslationError(RuntimeError):
    pass


class SubtitleTranslator:
    def __init__(
        self,
        llm: OpenAICompatibleLLM,
        config: LLMConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or LLMConfig()
        self.progress = progress

    def translate(
        self,
        items: list[SubtitleItem],
        target_language: str,
        source_language: str | None = None,
    ) -> list[SubtitleItem]:
        translated: list[SubtitleItem] = []
        batches = _batched(items, self.config.batch_size)
        for index, batch in enumerate(batches, start=1):
            _report(
                self.progress,
                f"translation batch {index}/{len(batches)} ({len(batch)} subtitle(s))",
            )
            translations = self._translate_batch(batch, target_language, source_language)
            translated.extend(
                item.with_translation(translations.get(item.id)) for item in batch
            )
        return translated

    def _translate_batch(
        self,
        items: list[SubtitleItem],
        target_language: str,
        source_language: str | None,
    ) -> dict[int, str]:
        payload = {
            "target_language": target_language,
            "source_language": source_language,
            "requirements": [
                "Translate the following subtitle lines as one coherent context.",
                "Return exactly one item for each id.",
                "Do not merge, split, remove, or reorder ids.",
                "Keep translations concise and suitable for subtitles.",
            ],
            "items": [{"id": item.id, "text": item.text} for item in items],
        }
        response = self.llm.complete_json(TRANSLATION_SYSTEM_PROMPT, payload)
        translations = validate_translation_output(response.parsed, [item.id for item in items])
        if translations is None:
            raise TranslationError("LLM translation output failed id validation.")
        return translations


def validate_translation_output(raw_output: Any, expected_ids: list[int]) -> dict[int, str] | None:
    if isinstance(raw_output, dict) and "items" in raw_output:
        raw_output = raw_output["items"]
    if not isinstance(raw_output, list) or len(raw_output) != len(expected_ids):
        return None

    translations: dict[int, str] = {}
    actual_ids: list[int] = []
    for item in raw_output:
        if not isinstance(item, dict):
            return None
        item_id = item.get("id")
        text = item.get("text")
        if not isinstance(item_id, int) or not isinstance(text, str):
            return None
        actual_ids.append(item_id)
        translations[item_id] = text.strip()

    if actual_ids != expected_ids:
        return None
    return translations


def _batched(items: list[SubtitleItem], batch_size: int) -> list[list[SubtitleItem]]:
    if batch_size <= 0:
        batch_size = len(items) or 1
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
