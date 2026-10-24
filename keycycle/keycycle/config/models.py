import os
from typing import Any, TypedDict

from .enums import RateLimitStrategy
from .loader import load_openrouter_models, load_rate_limits_from_yaml

# Get directory of this file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CONFIG_DIR, "models")


class ModelDict(TypedDict):
    """All the provider currrently supported"""

    cerebras: Any
    groq: Any
    gemini: Any
    openrouter: Any
    cohere: Any
    moonshot: Any
    mistral: Any


PROVIDER_STRATEGIES: ModelDict = {
    "cerebras": RateLimitStrategy.PER_MODEL,
    "groq": RateLimitStrategy.PER_MODEL,
    "gemini": RateLimitStrategy.PER_MODEL,
    "openrouter": RateLimitStrategy.GLOBAL,
    "cohere": RateLimitStrategy.PER_MODEL,
    "moonshot": RateLimitStrategy.PER_MODEL,
    "mistral": RateLimitStrategy.PER_MODEL,
}

# Load Cohere tiers first to handle the env var logic
_cohere_tiers = load_rate_limits_from_yaml(os.path.join(MODELS_DIR, "cohere.yaml"))

COHERE_TIERS = {
    "free": _cohere_tiers["free"],
    "pro": _cohere_tiers["pro"],
    "enterprise": _cohere_tiers["enterprise"],
}

MODEL_LIMITS: ModelDict = {
    "cerebras": load_rate_limits_from_yaml(os.path.join(MODELS_DIR, "cerebras.yaml")),
    "groq": load_rate_limits_from_yaml(os.path.join(MODELS_DIR, "groq.yaml")),
    "gemini": load_rate_limits_from_yaml(os.path.join(MODELS_DIR, "gemini.yaml")),
    "openrouter": load_rate_limits_from_yaml(os.path.join(MODELS_DIR, "openrouter.yaml")),
    "cohere": {
        # Same for every model
        "default": COHERE_TIERS.get(os.getenv("COHERE_TIER", "free").lower(), COHERE_TIERS["free"])
    },
    "moonshot": load_rate_limits_from_yaml(os.path.join(MODELS_DIR, "moonshot.yaml")),
    "mistral": load_rate_limits_from_yaml(os.path.join(MODELS_DIR, "mistral.yaml")),
}

OPENROUTER_MODELS = load_openrouter_models(os.path.join(MODELS_DIR, "openrouter_models.yaml"))
