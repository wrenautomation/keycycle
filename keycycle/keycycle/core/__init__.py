"""Core utilities and exceptions for keycycle."""

from .backoff import BackoffConfig, ExponentialBackoff
from .exceptions import (
    ConfigurationError,
    InvalidConfigurationError,
    InvalidKeyError,
    KeycycleError,
    KeycycleKeyError,
    KeyNotFoundError,
    MissingEnvironmentVariableError,
    NoAvailableKeyError,
    RateLimitError,
)
from .utils import (
    get_key_suffix,
    is_auth_error,
    is_dead_key_error,
    is_rate_limit_error,
    validate_api_key,
)

__all__ = [
    # Exceptions
    "KeycycleError",
    "KeycycleKeyError",
    "NoAvailableKeyError",
    "KeyNotFoundError",
    "InvalidKeyError",
    "RateLimitError",
    "ConfigurationError",
    "MissingEnvironmentVariableError",
    "InvalidConfigurationError",
    # Utilities
    "get_key_suffix",
    "is_rate_limit_error",
    "is_auth_error",
    "is_dead_key_error",
    "validate_api_key",
    # Backoff
    "ExponentialBackoff",
    "BackoffConfig",
]
