import pytest

import subtitle_gen.models as models


class FakeASR:
    instances = 0

    def __init__(self, *args, **kwargs):
        FakeASR.instances += 1


class FakeAligner:
    instances = 0

    def __init__(self, *args, **kwargs):
        FakeAligner.instances += 1


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeASR.instances = 0
    FakeAligner.instances = 0
    models.clear_model_cache()


def test_get_asr_returns_same_instance_for_same_config(monkeypatch):
    monkeypatch.setattr(models, "QwenASR", FakeASR)
    models.clear_model_cache()

    first = models.get_asr("model-a", device_map="auto", dtype="auto")
    second = models.get_asr("model-a", device_map="auto", dtype="auto")

    assert first is second
    assert FakeASR.instances == 1


def test_get_asr_loads_separately_for_different_config(monkeypatch):
    monkeypatch.setattr(models, "QwenASR", FakeASR)
    models.clear_model_cache()

    first = models.get_asr("model-a", device_map="auto", dtype="auto")
    second = models.get_asr("model-b", device_map="auto", dtype="auto")

    assert first is not second
    assert FakeASR.instances == 2


def test_get_aligner_reuses_instance(monkeypatch):
    monkeypatch.setattr(models, "QwenForcedAligner", FakeAligner)
    models.clear_model_cache()

    first = models.get_aligner("aligner-a")
    second = models.get_aligner("aligner-a")

    assert first is second
    assert FakeAligner.instances == 1


def test_clear_model_cache_drops_instances(monkeypatch):
    monkeypatch.setattr(models, "QwenASR", FakeASR)
    models.clear_model_cache()

    first = models.get_asr("model-a")
    models.clear_model_cache()
    second = models.get_asr("model-a")

    assert first is not second
    assert FakeASR.instances == 2
