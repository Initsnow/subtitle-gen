from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_ASR_MODEL
from .media import read_mono_wav_array


class ASRError(RuntimeError):
    pass


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str | None = None


class QwenASR:
    def __init__(
        self,
        model_id: str = DEFAULT_ASR_MODEL,
        device_map: str = "auto",
        dtype: str = "auto",
        compile_model: bool = False,
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.compile_model = compile_model
        self.max_new_tokens = max_new_tokens
        self._processor: Any | None = None
        self._model: Any | None = None
        self._compiled_forward: Any | None = None
        self._eager_forward: Any | None = None

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        processor, model = self._load()
        request: dict[str, Any] = {"audio": read_mono_wav_array(audio_path)}
        if language:
            request["language"] = language

        inputs = processor.apply_transcription_request(**request)
        inputs = _move_batch_to_model(inputs, model)

        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        except Exception:
            if self._compiled_forward is None or self._eager_forward is None:
                raise
            model.forward = self._eager_forward
            self._compiled_forward = None
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        if "input_ids" in inputs:
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        parsed = processor.decode(generated_ids, return_format="parsed")
        first = parsed[0] if isinstance(parsed, list) else parsed
        if isinstance(first, dict):
            return ASRResult(
                text=str(first.get("transcription") or first.get("text") or "").strip(),
                language=_optional_str(first.get("language")),
            )
        return ASRResult(text=str(first).strip(), language=language)

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ASRError(
                "Qwen ASR dependencies are not installed. Run `uv sync --extra models`."
            ) from exc

        dtype = _resolve_dtype(self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = _from_pretrained_with_dtype(
            AutoModelForMultimodalLM,
            self.model_id,
            dtype=dtype,
            device_map=self.device_map,
        )
        self._model.eval()
        if self.compile_model:
            self._compiled_forward, self._eager_forward = _try_compile_forward(self._model)
        return self._processor, self._model


def _resolve_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    try:
        import torch
    except ImportError as exc:
        raise ASRError("torch is required for model loading.") from exc
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ASRError(f"Unsupported dtype: {dtype}")
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
        raise ASRError("torch is required for inference.") from exc
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


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
