from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_ALIGNER_MODEL
from .media import read_mono_wav_array
from .types import TimedToken


class AlignmentError(RuntimeError):
    pass


class QwenForcedAligner:
    def __init__(
        self,
        model_id: str = DEFAULT_ALIGNER_MODEL,
        device_map: str = "auto",
        dtype: str = "auto",
        compile_model: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.compile_model = compile_model
        self._processor: Any | None = None
        self._model: Any | None = None
        self._compiled_forward: Any | None = None
        self._eager_forward: Any | None = None

    def align(
        self,
        audio_path: str | Path,
        transcript: str,
        language: str | None = None,
    ) -> list[TimedToken]:
        transcript = transcript.strip()
        if not transcript:
            return []

        processor, model = self._load()
        language = language or "English"
        aligner_inputs, word_lists = processor.prepare_forced_aligner_inputs(
            audio=read_mono_wav_array(audio_path),
            transcript=transcript,
            language=language,
        )
        aligner_inputs = _move_batch_to_model(aligner_inputs, model)

        try:
            import torch
        except ImportError as exc:
            raise AlignmentError("torch is required for forced alignment.") from exc

        with torch.inference_mode():
            try:
                outputs = model(**aligner_inputs)
            except Exception:
                if self._compiled_forward is None or self._eager_forward is None:
                    raise
                model.forward = self._eager_forward
                self._compiled_forward = None
                outputs = model(**aligner_inputs)

        decoded = processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )
        first = decoded[0] if isinstance(decoded, list) else decoded
        return [_coerce_timed_token(item) for item in first]

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoModelForTokenClassification, AutoProcessor
        except ImportError as exc:
            raise AlignmentError(
                "Qwen aligner dependencies are not installed. Run `uv sync --extra models`."
            ) from exc

        dtype = _resolve_dtype(self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = _from_pretrained_with_dtype(
            AutoModelForTokenClassification,
            self.model_id,
            dtype=dtype,
            device_map=self.device_map,
        )
        self._model.eval()
        if self.compile_model:
            self._compiled_forward, self._eager_forward = _try_compile_forward(self._model)
        return self._processor, self._model


def _coerce_timed_token(item: Any) -> TimedToken:
    if isinstance(item, dict):
        text = item.get("text") or item.get("word") or item.get("token")
        start = _first_present(item, "start", "start_time")
        end = _first_present(item, "end", "end_time")
    else:
        text = getattr(item, "text", None) or getattr(item, "word", None)
        start = _first_attr(item, "start", "start_time")
        end = _first_attr(item, "end", "end_time")
    if text is None or start is None or end is None:
        raise AlignmentError(f"Invalid forced alignment item: {item!r}")
    return TimedToken(text=str(text), start=float(start), end=float(end))


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _first_attr(item: Any, *names: str) -> Any:
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _resolve_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    try:
        import torch
    except ImportError as exc:
        raise AlignmentError("torch is required for model loading.") from exc
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise AlignmentError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _from_pretrained_with_dtype(model_cls: Any, model_id: str, dtype: Any, device_map: str) -> Any:
    kwargs = {"device_map": device_map}
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        return model_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return model_cls.from_pretrained(model_id, **kwargs)


def _move_batch_to_model(inputs: Any, model: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise AlignmentError("torch is required for inference.") from exc
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if hasattr(inputs, "to"):
        dtype = getattr(model, "dtype", None)
        if dtype is not None:
            return inputs.to(device, dtype)
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _try_compile_forward(model: Any) -> tuple[Any | None, Any | None]:
    try:
        import torch

        eager_forward = model.forward
        compiled_forward = torch.compile(eager_forward)
        model.forward = compiled_forward
        return compiled_forward, eager_forward
    except Exception:
        return None, None
