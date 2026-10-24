import os
from typing import Any, Optional, TypedDict

import yaml

from .dataclasses import RateLimits


class RateLimitConfig(TypedDict):
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    tokens_per_minute: Optional[int]
    tokens_per_hour: Optional[int]
    tokens_per_day: Optional[int]


class ModelConfig(TypedDict):
    name: str
    id: str
    max_context_length: int


def load_yaml_config(file_path: str) -> Any:
    """Loads a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rate_limits_from_yaml(file_path: str) -> dict[str, RateLimits]:
    """Loads rate limits from a YAML file and converts them to RateLimits objects."""
    data = load_yaml_config(file_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid rate limit config in {file_path}, expected a dictionary.")

    limits = {}
    for model_id, config in data.items():
        try:
            limits[model_id] = RateLimits(
                requests_per_minute=config["requests_per_minute"],
                requests_per_hour=config["requests_per_hour"],
                requests_per_day=config["requests_per_day"],
                tokens_per_minute=config.get("tokens_per_minute"),
                tokens_per_hour=config.get("tokens_per_hour"),
                tokens_per_day=config.get("tokens_per_day"),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field {e} for model {model_id} in {file_path}")

    return limits


def load_openrouter_models(file_path: str) -> list[ModelConfig]:
    """Loads the OpenRouter models list."""
    data = load_yaml_config(file_path)
    if not isinstance(data, list):
        raise ValueError(f"Invalid OpenRouter config in {file_path}, expected a list.")
    return data
