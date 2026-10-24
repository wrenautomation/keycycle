"""Custom exception hierarchy for keycycle."""

from typing import Any


class KeycycleError(Exception):
    """Base exception for all keycycle errors."""

    pass


class KeycycleKeyError(KeycycleError):
    """Base for key-related errors."""

    pass


class NoAvailableKeyError(KeycycleKeyError):
    """Raised when no API keys are available for use."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        wait: bool,
        timeout: float,
        total_keys: int = 0,
        cooling_down: int = 0,
    ):
        self.provider = provider
        self.model_id = model_id
        self.wait = wait
        self.timeout = timeout
        self.total_keys = total_keys
        self.cooling_down = cooling_down

        if wait:
            msg = f"Timeout: No available API keys for {provider}/{model_id} after {timeout}s"
        else:
            msg = f"No available API keys for {provider}/{model_id} (wait=False)"

        if total_keys > 0:
            msg += f" [{cooling_down}/{total_keys} keys cooling down]"

        super().__init__(msg)


class KeyNotFoundError(KeycycleKeyError):
    """Raised when a specific key identifier cannot be found."""

    def __init__(self, identifier: Any):
        self.identifier = identifier
        super().__init__(f"Key with identifier '{identifier}' not found.")


class InvalidKeyError(KeycycleKeyError):
    """Raised when an API key is invalid (401/403 response)."""

    def __init__(self, key_suffix: str, status_code: int):
        self.key_suffix = key_suffix
        self.status_code = status_code
        super().__init__(f"API key ...{key_suffix} is invalid (HTTP {status_code})")


class RateLimitError(KeycycleError):
    """Raised when rate limit is hit and retries exhausted."""

    def __init__(self, provider: str, model_id: str, attempts: int):
        self.provider = provider
        self.model_id = model_id
        self.attempts = attempts
        super().__init__(f"Rate limit exceeded for {provider}/{model_id} after {attempts} attempts")


class ConfigurationError(KeycycleError):
    """Raised for configuration-related errors."""

    pass


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when a required environment variable is missing."""

    def __init__(self, var_name: str):
        self.var_name = var_name
        super().__init__(f"Environment variable '{var_name}' not found.")


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(self, config_name: str, value: Any, expected: str):
        self.config_name = config_name
        self.value = value
        self.expected = expected
        super().__init__(f"'{config_name}' must be {expected}, got: {value}")
