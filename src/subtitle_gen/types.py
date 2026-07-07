from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimedToken:
    text: str
    start: float
    end: float

    def shifted(self, offset: float) -> "TimedToken":
        return TimedToken(text=self.text, start=self.start + offset, end=self.end + offset)


@dataclass(frozen=True)
class SubtitleItem:
    id: int
    start: float
    end: float
    text: str
    translation: str | None = None

    def with_translation(self, translation: str | None) -> "SubtitleItem":
        return SubtitleItem(
            id=self.id,
            start=self.start,
            end=self.end,
            text=self.text,
            translation=translation,
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }
        if self.translation is not None:
            data["translation"] = self.translation
        return data


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    start: float
    end: float
    text: str
    language: str | None = None
