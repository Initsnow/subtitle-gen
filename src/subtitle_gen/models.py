"""Process-wide model registry.

Loading the ASR, forced-aligner, and VAD weights dominates startup cost
(billions of parameters and tens of seconds). The registry below caches one
model wrapper per configuration so a process that handles many media files —
the batch CLI, or a library caller looping over inputs — loads each set of
weights exactly once instead of once per file.

Model wrappers are keyed by their full constructor configuration, so changing
the model id, device map, dtype, or compile flags transparently loads a
separate instance. ``clear_model_cache()`` drops every cached wrapper for
callers that need to reclaim the memory.
"""

from __future__ import annotations

from .aligner import QwenForcedAligner
from .asr import QwenASR

_asr_models: dict[tuple[str, str, str, bool, int], QwenASR] = {}
_aligner_models: dict[tuple[str, str, str, bool], QwenForcedAligner] = {}


def get_asr(
    model_id: str,
    device_map: str = "auto",
    dtype: str = "auto",
    compile_model: bool = False,
    max_new_tokens: int = 2048,
) -> QwenASR:
    """Return a shared :class:`QwenASR` for the given configuration.

    The first call constructs the wrapper; subsequent calls with identical
    parameters return the same instance so its lazily-loaded weights are
    reused across files.
    """
    key = (model_id, device_map, dtype, compile_model, max_new_tokens)
    model = _asr_models.get(key)
    if model is None:
        model = QwenASR(
            model_id=model_id,
            device_map=device_map,
            dtype=dtype,
            compile_model=compile_model,
            max_new_tokens=max_new_tokens,
        )
        _asr_models[key] = model
    return model


def get_aligner(
    model_id: str,
    device_map: str = "auto",
    dtype: str = "auto",
    compile_model: bool = True,
) -> QwenForcedAligner:
    """Return a shared :class:`QwenForcedAligner` for the given configuration."""
    key = (model_id, device_map, dtype, compile_model)
    model = _aligner_models.get(key)
    if model is None:
        model = QwenForcedAligner(
            model_id=model_id,
            device_map=device_map,
            dtype=dtype,
            compile_model=compile_model,
        )
        _aligner_models[key] = model
    return model


def clear_model_cache() -> None:
    """Drop every cached model wrapper.

    The wrappers only hold the weights lazily, so this releases GPU/CPU memory
    for configurations that will not be used again. Existing references held
    by callers keep working.
    """
    _asr_models.clear()
    _aligner_models.clear()


__all__ = ["clear_model_cache", "get_aligner", "get_asr"]
