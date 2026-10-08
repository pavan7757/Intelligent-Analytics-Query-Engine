import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=False)


def _current_gemini_model() -> str:
    return "gemini-3.5-flash"


def _normalize_model_name(model_name: str | None, provider: str) -> str:
    if provider not in {"gemini", "google", "google_gemini"}:
        return model_name or "gpt-4o-mini"
    default_model = _current_gemini_model()
    return (model_name or "").strip() or default_model


configured_dataset_dir = os.getenv("DATASET_DIR", "").strip()
if configured_dataset_dir:
    DEFAULT_DATASET_DIR = Path(configured_dataset_dir)
else:
    candidates = [
        BASE_DIR,
        BASE_DIR.parent,
        Path(__file__).resolve().parents[1].parent,
        Path.cwd(),
    ]
    expected_files = (
        "sales_data.csv", "targets.csv", "data_dictionary.json", "nl_queries.json"
    )
    DEFAULT_DATASET_DIR = Path("/dataset")
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in expected_files):
            DEFAULT_DATASET_DIR = candidate
            break

class Settings:
    dataset_dir: Path = DEFAULT_DATASET_DIR
    sales_file: str = "sales_data.csv"
    targets_file: str = "targets.csv"
    dictionary_file: str = "data_dictionary.json"
    queries_file: str = "nl_queries.json"
    feedback_file: str = "feedback_log.csv"
    llm_provider: str = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    llm_api_key: str | None = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or None
    )
    llm_model: str = _normalize_model_name(
        os.getenv("LLM_MODEL"),
        llm_provider,
    ) if llm_provider in {"gemini", "google", "google_gemini"} else (os.getenv("LLM_MODEL") or "gpt-4o-mini")
    llm_base_url: str = os.getenv("LLM_BASE_URL") or (
        "https://generativelanguage.googleapis.com" if llm_provider == "gemini" else "https://api.openai.com/v1"
    )
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY") or None
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "").strip()
    xai_api_key: str | None = os.getenv("XAI_API_KEY") or None
    xai_model: str = os.getenv("XAI_MODEL", "").strip()
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


settings = Settings()
