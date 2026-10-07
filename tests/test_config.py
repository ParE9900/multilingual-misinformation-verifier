"""Unit tests for configuration loader module."""

import os
from pathlib import Path
import pytest
from src.config import Config, get_config


def test_get_config_loads_from_existing_env():
    """Verify that get_config successfully loads keys from the project .env."""
    config = get_config()
    assert isinstance(config, Config)
    assert config.GROQ_API_KEY and len(config.GROQ_API_KEY) > 0
    assert config.GEMINI_API_KEY and len(config.GEMINI_API_KEY) > 0
    assert config.HF_TOKEN and len(config.HF_TOKEN) > 0
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"
    assert config.GEMINI_MODEL == "gemini-3.5-flash"
    assert config.NLI_MODEL_NAME == "xlm-roberta-base"
    assert config.LANG_DETECT_MODEL == "papluca/xlm-roberta-base-language-detection"
    assert config.CACHE_DIR == ".cache"


def test_get_config_raises_on_missing_keys(monkeypatch, tmp_path):
    """Verify RuntimeError is raised when any required API key is missing."""
    # Create empty dummy env file
    empty_env = tmp_path / ".env.empty"
    empty_env.write_text("", encoding="utf-8")

    # Remove relevant environment variables
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_config(env_path=empty_env, override=True)

    err_msg = str(exc_info.value)
    assert "Missing required API key(s)" in err_msg
    assert "GROQ_API_KEY" in err_msg
    assert "GEMINI_API_KEY" in err_msg
    assert "HF_TOKEN" in err_msg


def test_get_config_partial_missing_key(monkeypatch, tmp_path):
    """Verify RuntimeError is raised when one specific required key is missing."""
    custom_env = tmp_path / ".env.partial"
    custom_env.write_text(
        "GROQ_API_KEY=test_groq\nGEMINI_API_KEY=test_gemini\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_config(env_path=custom_env, override=True)

    assert "HF_TOKEN" in str(exc_info.value)
