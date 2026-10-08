import os


def test_default_gemini_model_is_current():
    original = os.environ.get("LLM_MODEL")
    os.environ.pop("LLM_MODEL", None)
    try:
        from app import config
        reload = __import__("importlib").reload
        reload(config)
        assert config.settings.llm_model == "gemini-3.5-flash"
    finally:
        if original is None:
            os.environ.pop("LLM_MODEL", None)
        else:
            os.environ["LLM_MODEL"] = original
        from app import config as cfg
        __import__("importlib").reload(cfg)
