from __future__ import annotations

import re
from types import TracebackType

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


_AUDIO_SUMMARY_RE = re.compile(
    r"^audio: (?P<duration>.+); chunks: (?P<chunks>\d+); media id: (?P<media_id>\S+)$"
)
_MODE_RE = re.compile(r"^mode: (?P<mode>[^;]+);")
_CHUNK_START_RE = re.compile(
    r"^chunk (?P<index>\d+)/(?P<total>\d+) (?P<window>.+)$"
)
_CHUNK_STATUS_RE = re.compile(
    r"^chunk (?P<index>\d+)/(?P<total>\d+): (?P<status>.+)$"
)
_LLM_REQUESTS_RE = re.compile(r"^LLM segmentation requests: (?P<total>\d+)$")
_LLM_RESULT_RE = re.compile(
    r"^LLM segmentation (?P<index>\d+)/(?P<total>\d+): (?P<status>.+)$"
)
_TRANSLATION_BATCH_RE = re.compile(
    r"^translation batch (?P<index>\d+)/(?P<total>\d+) "
    r"\((?P<count>\d+) subtitle\(s\)\)$"
)


class RichProgressReporter:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(stderr=True)
        self.progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[detail]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            console=self.console,
        )
        self._started = False
        self._segment_mode: str | None = None

        self._chunk_task: TaskID | None = None
        self._chunk_total = 0
        self._active_chunk: int | None = None
        self._completed_chunks: set[int] = set()

        self._llm_task: TaskID | None = None
        self._llm_total = 0
        self._completed_llm_requests: set[int] = set()
        self._llm_fallbacks: list[str] = []

        self._translation_task: TaskID | None = None
        self._translation_total = 0

    def __enter__(self) -> RichProgressReporter:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_type is None:
            self._complete_all_tasks()
        self.stop()
        return False

    def __call__(self, message: str) -> None:
        self.report(message)

    def start(self) -> None:
        if not self._started:
            self.progress.start()
            self._started = True

    def stop(self) -> None:
        if self._started:
            self.progress.stop()
            self._started = False

    def report(self, message: str) -> None:
        self.start()

        if self._handle_mode(message):
            return
        if self._handle_audio_summary(message):
            return
        if self._handle_chunk_start(message):
            return
        if self._handle_chunk_status(message):
            return
        if self._handle_llm_requests(message):
            return
        if self._handle_llm_result(message):
            return
        if self._handle_translation_batch(message):
            return
        if self._handle_milestone(message):
            return

        self._print_info(message)

    def _handle_mode(self, message: str) -> bool:
        match = _MODE_RE.match(message)
        if not match:
            return False
        self._segment_mode = match.group("mode")
        self._print_info(f"mode: {message.removeprefix('mode: ')}")
        return True

    def _handle_audio_summary(self, message: str) -> bool:
        match = _AUDIO_SUMMARY_RE.match(message)
        if not match:
            return False
        total = int(match.group("chunks"))
        self._chunk_total = total
        if total > 0 and self._chunk_task is None:
            self._chunk_task = self.progress.add_task(
                "Chunks",
                total=total,
                detail=f"0/{total}",
            )
        self._print_info(
            "audio: "
            f"{match.group('duration')}; chunks: {total}; "
            f"media id: {match.group('media_id')}"
        )
        return True

    def _handle_chunk_start(self, message: str) -> bool:
        match = _CHUNK_START_RE.match(message)
        if not match:
            return False

        index = int(match.group("index"))
        total = int(match.group("total"))
        self._ensure_chunk_task(total)
        self._complete_active_chunk_before(index)
        self._active_chunk = index
        self._update_chunk_detail(f"{index}/{total} {match.group('window')}")
        return True

    def _handle_chunk_status(self, message: str) -> bool:
        match = _CHUNK_STATUS_RE.match(message)
        if not match:
            return False

        index = int(match.group("index"))
        total = int(match.group("total"))
        status = match.group("status")
        self._ensure_chunk_task(total)
        self._complete_active_chunk_before(index)
        self._active_chunk = index
        self._update_chunk_detail(f"{index}/{total} {status}")
        if self._is_final_chunk_status(status):
            self._complete_chunk(index)
        return True

    def _handle_llm_requests(self, message: str) -> bool:
        match = _LLM_REQUESTS_RE.match(message)
        if not match:
            return False

        total = int(match.group("total"))
        self._llm_total = total
        self._completed_llm_requests.clear()
        self._llm_fallbacks.clear()
        if total <= 0:
            self._print_info("LLM segmentation: no requests")
            return True
        self._llm_task = self.progress.add_task(
            "LLM segmentation",
            total=total,
            detail=f"0/{total}",
        )
        return True

    def _handle_llm_result(self, message: str) -> bool:
        match = _LLM_RESULT_RE.match(message)
        if not match:
            return False

        index = int(match.group("index"))
        total = int(match.group("total"))
        status = match.group("status")
        self._ensure_llm_task(total)
        if status.startswith("fallback"):
            self._llm_fallbacks.append(f"{index}/{total}: {status}")
        self._completed_llm_requests.add(index)
        self.progress.update(
            self._llm_task,
            completed=len(self._completed_llm_requests),
            detail=escape(f"{index}/{total} {status}"),
        )
        if len(self._completed_llm_requests) >= total and self._llm_fallbacks:
            last = self._llm_fallbacks[-1]
            self.console.print(
                "[yellow]subtitle-gen:[/] "
                f"{len(self._llm_fallbacks)}/{total} LLM segmentation fallback(s); "
                f"last {escape(last)}"
            )
        return True

    def _handle_translation_batch(self, message: str) -> bool:
        match = _TRANSLATION_BATCH_RE.match(message)
        if not match:
            return False

        index = int(match.group("index"))
        total = int(match.group("total"))
        count = int(match.group("count"))
        if self._translation_task is None:
            self._translation_task = self.progress.add_task(
                "Translation",
                total=total,
                detail=f"0/{total}",
            )
        self._translation_total = total
        self.progress.update(
            self._translation_task,
            completed=max(0, index - 1),
            detail=escape(f"{index}/{total} ({count} subtitle(s))"),
        )
        return True

    def _handle_milestone(self, message: str) -> bool:
        if message.startswith("segmenting with "):
            self._complete_all_chunks()
            self._print_info(message)
            return True
        if message.startswith("hybrid soft split:"):
            self._print_info(message)
            return True
        if message.startswith("hybrid hard fallback:"):
            self.console.print(f"[yellow]subtitle-gen:[/] {escape(message)}")
            return True
        if message.startswith("subtitles:"):
            self._complete_all_chunks()
            self._complete_llm_task()
            self._print_info(message)
            return True
        if message.startswith("notice:"):
            self.console.print(f"[yellow]subtitle-gen:[/] {escape(message)}")
            return True
        if message == "translation complete":
            self._complete_translation_task()
            self._print_info(message)
            return True
        if message.startswith("wrote "):
            self._print_info(message)
            return True
        return False

    def _print_info(self, message: str) -> None:
        self.console.print(f"[green]subtitle-gen:[/] {escape(message)}")

    def _ensure_chunk_task(self, total: int) -> None:
        if total <= 0:
            return
        self._chunk_total = max(self._chunk_total, total)
        if self._chunk_task is None:
            self._chunk_task = self.progress.add_task(
                "Chunks",
                total=self._chunk_total,
                detail=f"0/{self._chunk_total}",
            )
        else:
            self.progress.update(self._chunk_task, total=self._chunk_total)

    def _update_chunk_detail(self, detail: str) -> None:
        if self._chunk_task is not None:
            self.progress.update(self._chunk_task, detail=escape(detail))

    def _complete_active_chunk_before(self, index: int) -> None:
        if self._active_chunk is None or self._active_chunk >= index:
            return
        self._complete_chunk(self._active_chunk)

    def _complete_chunk(self, index: int) -> None:
        if self._chunk_task is None or index in self._completed_chunks:
            return
        self._completed_chunks.add(index)
        completed = min(len(self._completed_chunks), self._chunk_total)
        self.progress.update(self._chunk_task, completed=completed)

    def _complete_all_chunks(self) -> None:
        if self._chunk_task is None or self._chunk_total <= 0:
            return
        self._completed_chunks.update(range(1, self._chunk_total + 1))
        self.progress.update(self._chunk_task, completed=self._chunk_total)

    def _is_final_chunk_status(self, status: str) -> bool:
        if "alignment cache hit" in status:
            return True
        return self._segment_mode == "none" and "ASR cache hit" in status

    def _ensure_llm_task(self, total: int) -> None:
        self._llm_total = max(self._llm_total, total)
        if self._llm_task is None:
            self._llm_task = self.progress.add_task(
                "LLM segmentation",
                total=self._llm_total,
                detail=f"0/{self._llm_total}",
            )
        else:
            self.progress.update(self._llm_task, total=self._llm_total)

    def _complete_llm_task(self) -> None:
        if self._llm_task is not None and self._llm_total > 0:
            self.progress.update(self._llm_task, completed=self._llm_total)

    def _complete_translation_task(self) -> None:
        if self._translation_task is not None and self._translation_total > 0:
            self.progress.update(
                self._translation_task,
                completed=self._translation_total,
                detail=escape(f"{self._translation_total}/{self._translation_total}"),
            )

    def _complete_all_tasks(self) -> None:
        self._complete_all_chunks()
        self._complete_llm_task()
        self._complete_translation_task()
