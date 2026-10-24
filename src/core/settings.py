"""Settings from the environment (.env). Keys are never hard-coded or committed; see .env.example."""
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    llm_model: str
    openai_api_key: str | None
    langsmith_tracing: bool


@lru_cache
def settings() -> Settings:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return Settings(llm_model=os.getenv("HR_LLM_MODEL", "gpt-4.1-mini"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true")
