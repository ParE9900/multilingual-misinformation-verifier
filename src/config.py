"""Configuration loader for the verifier project."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Config:
    """Project credentials and model configurations."""

    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    HF_TOKEN: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-3.5-flash"
    NLI_MODEL_NAME: str = "xlm-roberta-base"
    LANG_DETECT_MODEL: str = "papluca/xlm-roberta-base-language-detection"
    CACHE_DIR: str = ".cache"


def get_config(env_path: Optional[str | Path] = None, override: bool = False) -> Config:
    """Load configuration from environment variables or .env file."""
    if env_path is not None:
        dotenv_file = Path(env_path)
    else:
        dotenv_file = Path(__file__).resolve().parent.parent / ".env"
        if not dotenv_file.exists():
            dotenv_file = Path(".env")

    if dotenv_file.exists():
        load_dotenv(dotenv_path=dotenv_file, override=override)
    else:
        load_dotenv(override=override)

    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    missing = []
    if not groq_api_key:
        missing.append("GROQ_API_KEY")
    if not gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not hf_token:
        missing.append("HF_TOKEN")

    if missing:
        raise RuntimeError(f"Missing required API key(s): {', '.join(missing)}.")

    return Config(
        GROQ_API_KEY=groq_api_key,
        GEMINI_API_KEY=gemini_api_key,
        HF_TOKEN=hf_token,
        GROQ_MODEL=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        GEMINI_MODEL=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        NLI_MODEL_NAME=os.getenv("NLI_MODEL_NAME", "xlm-roberta-base"),
        LANG_DETECT_MODEL=os.getenv("LANG_DETECT_MODEL", "papluca/xlm-roberta-base-language-detection"),
        CACHE_DIR=os.getenv("CACHE_DIR", ".cache"),
    )
