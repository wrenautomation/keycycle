from .adapters.generic_adapter import (
    AsyncGenericRotatingClient,
    GenericClientConfig,
    SyncGenericRotatingClient,
    create_rotating_client,
    detect_async_client,
    get_valid_constructor_kwargs,
)
from .config.dataclasses import KeyLimitOverride, RateLimits
from .core.exceptions import (
    ConfigurationError,
    InvalidKeyError,
    KeycycleError,
    KeyNotFoundError,
    NoAvailableKeyError,
    RateLimitError,
)
from .core.utils import KeyEntry
from .key_rotation.rotation_manager import RotatingKeyManager
from .legacy_multi_provider_wrapper import MultiProviderWrapper
from .multi_client_wrapper import MultiClientWrapper, ProviderEnvConfig

__all__ = [
    # New primary wrapper (multi-provider support)
    "MultiClientWrapper",
    "ProviderEnvConfig",
    # Legacy wrapper (single provider, backward compatible)
    "MultiProviderWrapper",
    # Main classes
    "RateLimits",
    "KeyLimitOverride",
    "KeyEntry",
    "RotatingKeyManager",
    # Generic rotating client
    "create_rotating_client",
    "detect_async_client",
    "get_valid_constructor_kwargs",
    "GenericClientConfig",
    "SyncGenericRotatingClient",
    "AsyncGenericRotatingClient",
    # Exceptions
    "KeycycleError",
    "NoAvailableKeyError",
    "KeyNotFoundError",
    "InvalidKeyError",
    "RateLimitError",
    "ConfigurationError",
]
