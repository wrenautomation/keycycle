from enum import Enum


class RateLimitStrategy(Enum):
    PER_MODEL = "per_model"  # Cerebras, Groq, Gemini
    GLOBAL = "global"  # OpenRouter (Shared limits across all models)
