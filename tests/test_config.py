import pytest

from subtitle_gen.config import ConfigError, DEFAULT_ASR_MODEL, AppConfig, apply_overrides, load_config


def test_load_config_uses_default_values_without_config_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.model.asr_model == DEFAULT_ASR_MODEL
    assert config.segment.mode == "blingfire"


def test_load_config_reads_root_config_toml_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[model]
asr_model = "example/asr"

[llm]
concurrency = 4
""".strip(),
        encoding="utf-8",
    )

    config = load_config()

    assert config.model.asr_model == "example/asr"
    assert config.llm.concurrency == 4


def test_load_config_reads_llm_api_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "secret"
""".strip(),
        encoding="utf-8",
    )

    config = load_config()

    assert config.llm.api_key == "secret"


def test_load_config_reads_cache_cleanup_options(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[cache]
enabled = false
cleanup_enabled = false
max_media_entries = 3
""".strip(),
        encoding="utf-8",
    )

    config = load_config()

    assert config.cache.enabled is False
    assert config.cache.cleanup_enabled is False
    assert config.cache.max_media_entries == 3


def test_apply_overrides_can_disable_cache():
    config = apply_overrides(AppConfig(), cache_enabled=False)

    assert config.cache.enabled is False


def test_load_config_rejects_removed_llm_enabled_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[segment]
llm_enabled = false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config()


def test_load_config_rejects_out_of_range_llm_temperature(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        """
[llm]
temperature = 5.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=r"\[llm\]\.temperature"):
        load_config()


def test_apply_overrides_rejects_invalid_llm_concurrency():
    with pytest.raises(ConfigError, match=r"\[llm\]\.concurrency"):
        apply_overrides(AppConfig(), llm_concurrency=0)
