"""
Generic rotating client adapter for any Python client that accepts api_key= in its constructor.

This module provides a drop-in replacement wrapper that automatically rotates API keys
on rate limit errors for any compatible client library (Anthropic, TwelveLabs, Cohere, etc.).
"""

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
)

from ..config.constants import (
    KEY_ROTATION_DELAY_SECONDS,
    TEMP_RATE_LIMIT_INITIAL_DELAY,
    TEMP_RATE_LIMIT_MAX_DELAY,
    TEMP_RATE_LIMIT_MAX_RETRIES,
    TEMP_RATE_LIMIT_MULTIPLIER,
)
from ..config.dataclasses import KeyUsage, RateLimits
from ..core.backoff import BackoffConfig, ExponentialBackoff
from ..core.utils import (
    get_key_suffix,
    is_dead_key_error,
    is_rate_limit_error,
    is_temporary_rate_limit_error,
)
from ..key_rotation.rotation_manager import RotatingKeyManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


def detect_async_client(client_class: type) -> bool:
    """
    Auto-detect if a client class is async.

    Checks for:
    - 'Async' in class name
    - __aenter__, __aexit__, aclose methods
    - Public methods that are coroutine functions

    Args:
        client_class: The client class to check

    Returns:
        True if the client appears to be async, False otherwise
    """
    # Check class name for 'Async'
    if "Async" in client_class.__name__:
        return True

    # Check for async context manager methods
    if hasattr(client_class, "__aenter__") or hasattr(client_class, "__aexit__"):
        return True

    # Check for aclose method (common in async clients)
    if hasattr(client_class, "aclose"):
        aclose = getattr(client_class, "aclose")
        if asyncio.iscoroutinefunction(aclose):
            return True

    # Check if any public methods are coroutine functions
    for name in dir(client_class):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(client_class, name)
            if asyncio.iscoroutinefunction(attr):
                return True
        except (AttributeError, TypeError):
            continue

    return False


def default_usage_extractor(response: Any) -> int:
    """
    Default usage extractor that handles multiple common patterns.

    Supports:
    - response.usage.total_tokens (OpenAI)
    - response.usage.input_tokens + output_tokens (Anthropic)
    - response.meta.billed_units (Cohere)
    - response['usage']['total_tokens'] (dict-style)

    Args:
        response: The API response object

    Returns:
        Total tokens used, or 0 if unable to extract
    """
    try:
        # OpenAI style: response.usage.total_tokens
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            if hasattr(usage, "total_tokens") and usage.total_tokens is not None:
                return usage.total_tokens
            # Anthropic style: input_tokens + output_tokens
            if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                input_t = usage.input_tokens or 0
                output_t = usage.output_tokens or 0
                return input_t + output_t
    except (AttributeError, TypeError):
        pass

    # Cohere style: response.meta.billed_units
    try:
        if hasattr(response, "meta") and response.meta:
            meta = response.meta
            if hasattr(meta, "billed_units") and meta.billed_units:
                units = meta.billed_units
                total = 0
                if hasattr(units, "input_tokens"):
                    total += units.input_tokens or 0
                if hasattr(units, "output_tokens"):
                    total += units.output_tokens or 0
                return total
    except (AttributeError, TypeError):
        pass

    # Dict-style: response['usage']['total_tokens']
    try:
        if isinstance(response, dict):
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                total = usage.get("total_tokens")
                if total is not None:
                    return total
                # Dict with input/output tokens
                input_t = usage.get("input_tokens", 0) or 0
                output_t = usage.get("output_tokens", 0) or 0
                if input_t or output_t:
                    return input_t + output_t
    except (KeyError, TypeError):
        pass

    return 0


def get_valid_constructor_kwargs(client_class: type) -> Optional[frozenset[str]]:
    """
    Inspect client class to determine valid constructor kwargs.

    Returns:
        FrozenSet of valid kwarg names, or None if client accepts **kwargs
        (meaning introspection can't filter).
    """
    try:
        sig = inspect.signature(client_class.__init__)
    except (ValueError, TypeError):
        return None

    params = sig.parameters
    valid_kwargs = set()

    for name, param in params.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return None  # Accepts **kwargs, can't filter
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            valid_kwargs.add(name)

    return frozenset(valid_kwargs) if valid_kwargs else None


@dataclass
class GenericClientConfig:
    """Configuration for a generic rotating client."""

    client_class: type
    """The client class to wrap (e.g., Anthropic, TwelveLabs)"""

    api_key_param: str = "api_key"
    """Name of the API key parameter in the client constructor"""

    is_async: Optional[bool] = None
    """Whether the client is async. If None, auto-detected."""

    usage_extractor: Optional[Callable[[Any], int]] = None
    """Function to extract token usage from responses. Uses default if None."""

    estimated_tokens: int = 1000
    """Estimated tokens per request for rate limiting"""

    max_retries: int = 5
    """Maximum number of key rotations on rate limit errors"""

    model_param: str = "model"
    """Name of the model parameter in API calls"""

    client_kwargs: dict = field(default_factory=dict)
    """Additional kwargs to pass to the client constructor"""

    excluded_kwargs: frozenset[str] = field(default_factory=frozenset)
    """Kwargs to explicitly exclude from client constructor (manual exclusion layer)"""

    valid_kwargs: Optional[frozenset[str]] = None
    """Valid constructor kwargs from introspection. None means accept all (client uses **kwargs)."""


class BaseGenericRotatingClient(Generic[T]):
    """Base class for generic rotating clients."""

    def __init__(
        self,
        manager: RotatingKeyManager,
        limit_resolver: Callable[[str, Optional[str]], RateLimits],
        default_model: str,
        config: GenericClientConfig,
    ):
        """
        Initialize the base rotating client.

        Args:
            manager: The key rotation manager
            limit_resolver: Function to resolve rate limits for a model
            default_model: Default model to use
            config: Client configuration
        """
        self.manager = manager
        self.limit_resolver = limit_resolver
        self.default_model = default_model
        self.config = config
        self._usage_extractor = config.usage_extractor or default_usage_extractor

    def _get_fresh_client(self, key_usage: KeyUsage) -> T:
        """Create a fresh client instance with the key's params."""
        key_params = key_usage.get_client_params()

        # Merge client_kwargs with key_params (key_params take precedence)
        merged_kwargs = {**self.config.client_kwargs, **key_params}

        # Layer 1: Apply manual exclusions
        if self.config.excluded_kwargs:
            merged_kwargs = {
                k: v for k, v in merged_kwargs.items() if k not in self.config.excluded_kwargs
            }

        # Layer 2: Apply introspection filter (if available)
        if self.config.valid_kwargs is not None:
            final_kwargs = {k: v for k, v in merged_kwargs.items() if k in self.config.valid_kwargs}
            # Log filtered kwargs for debugging
            filtered = set(merged_kwargs.keys()) - set(final_kwargs.keys())
            if filtered:
                logger.debug(
                    "Introspection filtered kwargs for %s: %s",
                    self.config.client_class.__name__,
                    filtered,
                )
        else:
            final_kwargs = merged_kwargs

        return self.config.client_class(**final_kwargs)

    def _record_usage(
        self,
        key_usage: KeyUsage,
        model_id: str,
        actual_tokens: int,
        track_usage: Optional[bool] = None,
    ) -> None:
        """Record usage for a key."""
        self.manager.record_usage(
            key_obj=key_usage,
            model_id=model_id,
            actual_tokens=actual_tokens,
            estimated_tokens=self.config.estimated_tokens,
            track_usage=track_usage,
        )

    def _extract_usage(self, response: Any) -> int:
        """Extract token usage from a response."""
        return self._usage_extractor(response)

    def _get_model_id(self, kwargs: dict) -> str:
        """Extract model ID from kwargs or use default."""
        return kwargs.get(self.config.model_param, self.default_model)


class SyncGenericRotatingClient(BaseGenericRotatingClient[T]):
    """Synchronous generic rotating client."""

    def __getattr__(self, name: str) -> "SyncGenericProxyHelper":
        return SyncGenericProxyHelper(self, [name])

    def _execute(self, path: list[str], args: tuple, kwargs: dict) -> Any:
        """Execute a method call with key rotation."""
        # Per-call tracking override - popped before the kwargs reach the real
        # client, which may reject unknown keyword arguments.
        track_usage = kwargs.pop("track_usage", None)
        model_id = self._get_model_id(kwargs)
        limits = self.limit_resolver(model_id, None)

        for attempt in range(self.config.max_retries + 1):
            key_usage = self.manager.acquire_key(model_id, limits, self.config.estimated_tokens)
            if not key_usage:
                raise RuntimeError(self.manager.unavailable_message(model_id))

            # Create backoff for temporary rate limits
            temp_backoff = ExponentialBackoff(
                BackoffConfig(
                    initial_interval=TEMP_RATE_LIMIT_INITIAL_DELAY,
                    max_interval=TEMP_RATE_LIMIT_MAX_DELAY,
                    multiplier=TEMP_RATE_LIMIT_MULTIPLIER,
                )
            )

            for temp_attempt in range(TEMP_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    real_client = self._get_fresh_client(key_usage)

                    # Navigate to the target method
                    target = real_client
                    for p in path:
                        target = getattr(target, p)

                    result = target(*args, **kwargs)

                    # Handle streaming responses
                    if hasattr(result, "__iter__") and not isinstance(
                        result, (str, bytes, dict, list)
                    ):
                        # Check if it looks like a generator/iterator
                        if hasattr(result, "__next__") or inspect.isgenerator(result):
                            return self._wrap_stream(result, key_usage, model_id, track_usage)

                    self._record_usage(
                        key_usage, model_id, self._extract_usage(result), track_usage
                    )
                    return result

                except Exception as e:
                    # Check for temporary rate limit first - retry with SAME key
                    if (
                        is_temporary_rate_limit_error(e)
                        and temp_attempt < TEMP_RATE_LIMIT_MAX_RETRIES
                    ):
                        delay = temp_backoff.get_next_interval()
                        logger.info(
                            "Temporary rate limit on key ...%s for %s. Waiting %.1fs (%d/%d).",
                            get_key_suffix(key_usage.api_key),
                            model_id,
                            delay,
                            temp_attempt + 1,
                            TEMP_RATE_LIMIT_MAX_RETRIES,
                        )
                        time.sleep(delay)
                        continue  # Retry with SAME key

                    # Dead key (401/403 revoked/denied, 402 credits exhausted) - key is dead
                    # for this process, rotate to next key
                    if is_dead_key_error(e) and attempt < self.config.max_retries:
                        logger.warning(
                            "Dead key (401/402/403) hit for %s on key ...%s. Marking key dead and rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.config.max_retries + 1,
                        )
                        key_usage.mark_dead()
                        self.manager.force_rotate_index()
                        time.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    # Hard rate limit - rotate to next key
                    if is_rate_limit_error(e) and attempt < self.config.max_retries:
                        logger.warning(
                            "429/RateLimit hit for %s on key ...%s. Rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.config.max_retries + 1,
                        )
                        key_usage.trigger_cooldown()
                        self.manager.force_rotate_index()
                        time.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    self._record_usage(key_usage, model_id, 0, track_usage)
                    raise
            else:
                # Inner loop exhausted without success - continue to next key
                if attempt < self.config.max_retries:
                    logger.warning(
                        "Temporary rate limit retries exhausted for key ...%s. Rotating.",
                        get_key_suffix(key_usage.api_key),
                    )
                    key_usage.trigger_cooldown()
                    self.manager.force_rotate_index()
                    continue

        raise RuntimeError(f"All retry attempts exhausted for {model_id}")

    def _wrap_stream(
        self,
        generator: Generator,
        key_usage: KeyUsage,
        model_id: str,
        track_usage: Optional[bool] = None,
    ) -> Generator:
        """Wrap a streaming response to track usage and handle errors."""
        final_tokens = 0
        try:
            for chunk in generator:
                # Try to extract usage from streaming chunks
                chunk_tokens = self._extract_usage(chunk)
                if chunk_tokens:
                    final_tokens = chunk_tokens
                yield chunk
        except Exception as e:
            if is_dead_key_error(e):
                logger.warning(
                    "Dead key (401/402/403) hit during streaming for %s on key ...%s. Marking key dead.",
                    model_id,
                    get_key_suffix(key_usage.api_key),
                )
                key_usage.mark_dead()
                self.manager.force_rotate_index()
            elif is_rate_limit_error(e):
                logger.warning(
                    "Rate limit hit during streaming for %s on key ...%s.",
                    model_id,
                    get_key_suffix(key_usage.api_key),
                )
                key_usage.trigger_cooldown()
                self.manager.force_rotate_index()
            raise
        finally:
            self._record_usage(key_usage, model_id, final_tokens, track_usage)


class SyncGenericProxyHelper:
    """Helper class to build attribute path chains for sync clients."""

    def __init__(self, client: SyncGenericRotatingClient, path: list[str]):
        self.client = client
        self.path = path

    def __getattr__(self, name: str) -> "SyncGenericProxyHelper":
        return SyncGenericProxyHelper(self.client, self.path + [name])

    def __call__(self, *args, **kwargs) -> Any:
        return self.client._execute(self.path, args, kwargs)


class AsyncGenericRotatingClient(BaseGenericRotatingClient[T]):
    """Asynchronous generic rotating client."""

    def __getattr__(self, name: str) -> "AsyncGenericProxyHelper":
        return AsyncGenericProxyHelper(self, [name])

    async def _execute(self, path: list[str], args: tuple, kwargs: dict) -> Any:
        """Execute a method call with key rotation (async)."""
        # Per-call tracking override - popped before the kwargs reach the real
        # client, which may reject unknown keyword arguments.
        track_usage = kwargs.pop("track_usage", None)
        model_id = self._get_model_id(kwargs)
        limits = self.limit_resolver(model_id, None)

        for attempt in range(self.config.max_retries + 1):
            key_usage = await self.manager.acquire_key_async(
                model_id, limits, self.config.estimated_tokens
            )
            if not key_usage:
                raise RuntimeError(self.manager.unavailable_message(model_id))

            # Create backoff for temporary rate limits
            temp_backoff = ExponentialBackoff(
                BackoffConfig(
                    initial_interval=TEMP_RATE_LIMIT_INITIAL_DELAY,
                    max_interval=TEMP_RATE_LIMIT_MAX_DELAY,
                    multiplier=TEMP_RATE_LIMIT_MULTIPLIER,
                )
            )

            for temp_attempt in range(TEMP_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    real_client = self._get_fresh_client(key_usage)

                    # Navigate to the target method
                    target = real_client
                    for p in path:
                        target = getattr(target, p)

                    result = await target(*args, **kwargs)

                    # Handle async streaming responses
                    if hasattr(result, "__aiter__"):
                        return self._wrap_stream(result, key_usage, model_id, track_usage)

                    self._record_usage(
                        key_usage, model_id, self._extract_usage(result), track_usage
                    )
                    return result

                except Exception as e:
                    # Check for temporary rate limit first - retry with SAME key
                    if (
                        is_temporary_rate_limit_error(e)
                        and temp_attempt < TEMP_RATE_LIMIT_MAX_RETRIES
                    ):
                        delay = temp_backoff.get_next_interval()
                        logger.info(
                            "Temporary rate limit on key ...%s for %s. Waiting %.1fs (%d/%d).",
                            get_key_suffix(key_usage.api_key),
                            model_id,
                            delay,
                            temp_attempt + 1,
                            TEMP_RATE_LIMIT_MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue  # Retry with SAME key

                    # Dead key (401/403 revoked/denied, 402 credits exhausted) - key is dead
                    # for this process, rotate to next key
                    if is_dead_key_error(e) and attempt < self.config.max_retries:
                        logger.warning(
                            "Dead key (401/402/403) hit for %s on key ...%s. Marking key dead and rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.config.max_retries + 1,
                        )
                        key_usage.mark_dead()
                        self.manager.force_rotate_index()
                        await asyncio.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    # Hard rate limit - rotate to next key
                    if is_rate_limit_error(e) and attempt < self.config.max_retries:
                        logger.warning(
                            "429/RateLimit hit for %s on key ...%s. Rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.config.max_retries + 1,
                        )
                        key_usage.trigger_cooldown()
                        self.manager.force_rotate_index()
                        await asyncio.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    self._record_usage(key_usage, model_id, 0, track_usage)
                    raise
            else:
                # Inner loop exhausted without success - continue to next key
                if attempt < self.config.max_retries:
                    logger.warning(
                        "Temporary rate limit retries exhausted for key ...%s. Rotating.",
                        get_key_suffix(key_usage.api_key),
                    )
                    key_usage.trigger_cooldown()
                    self.manager.force_rotate_index()
                    continue

        raise RuntimeError(f"All retry attempts exhausted for {model_id}")

    async def _wrap_stream(
        self,
        generator: AsyncGenerator,
        key_usage: KeyUsage,
        model_id: str,
        track_usage: Optional[bool] = None,
    ) -> AsyncGenerator:
        """Wrap an async streaming response to track usage and handle errors."""
        final_tokens = 0
        try:
            async for chunk in generator:
                # Try to extract usage from streaming chunks
                chunk_tokens = self._extract_usage(chunk)
                if chunk_tokens:
                    final_tokens = chunk_tokens
                yield chunk
        except Exception as e:
            if is_dead_key_error(e):
                logger.warning(
                    "Dead key (401/402/403) hit during streaming for %s on key ...%s. Marking key dead.",
                    model_id,
                    get_key_suffix(key_usage.api_key),
                )
                key_usage.mark_dead()
                self.manager.force_rotate_index()
            elif is_rate_limit_error(e):
                logger.warning(
                    "Rate limit hit during streaming for %s on key ...%s.",
                    model_id,
                    get_key_suffix(key_usage.api_key),
                )
                key_usage.trigger_cooldown()
                self.manager.force_rotate_index()
            raise
        finally:
            self._record_usage(key_usage, model_id, final_tokens, track_usage)


class AsyncGenericProxyHelper:
    """Helper class to build attribute path chains for async clients."""

    def __init__(self, client: AsyncGenericRotatingClient, path: list[str]):
        self.client = client
        self.path = path

    def __getattr__(self, name: str) -> "AsyncGenericProxyHelper":
        return AsyncGenericProxyHelper(self.client, self.path + [name])

    async def __call__(self, *args, **kwargs) -> Any:
        return await self.client._execute(self.path, args, kwargs)


def create_rotating_client(
    client_class: type[T],
    manager: RotatingKeyManager,
    limit_resolver: Callable[[str, Optional[str]], RateLimits],
    default_model: str,
    api_key_param: str = "api_key",
    is_async: Optional[bool] = None,
    usage_extractor: Optional[Callable[[Any], int]] = None,
    estimated_tokens: int = 1000,
    max_retries: int = 5,
    model_param: str = "model",
    excluded_kwargs: Optional[list[str]] = None,
    **client_kwargs,
) -> Union[SyncGenericRotatingClient[T], AsyncGenericRotatingClient[T]]:
    """
    Factory function to create an appropriate rotating client wrapper.

    Args:
        client_class: The client class to wrap (e.g., Anthropic, TwelveLabs)
        manager: The key rotation manager
        limit_resolver: Function to resolve rate limits for a model
        default_model: Default model to use
        api_key_param: Name of the API key parameter in the client constructor
        is_async: Whether the client is async. If None, auto-detected
        usage_extractor: Function to extract token usage from responses
        estimated_tokens: Estimated tokens per request for rate limiting
        max_retries: Maximum number of key rotations on rate limit errors
        model_param: Name of the model parameter in API calls
        excluded_kwargs: List of kwarg names to explicitly exclude from client constructor
        **client_kwargs: Additional kwargs to pass to the client constructor

    Returns:
        A rotating client wrapper (sync or async based on client type)

    Example:
        >>> from anthropic import Anthropic
        >>> client = create_rotating_client(
        ...     Anthropic,
        ...     manager=wrapper.manager,
        ...     limit_resolver=wrapper._resolve_limits,
        ...     default_model="claude-3-sonnet",
        ... )
        >>> response = client.messages.create(...)

        >>> # With excluded_kwargs for clients that don't accept certain params
        >>> from twelvelabs import TwelveLabs
        >>> client = create_rotating_client(
        ...     TwelveLabs,
        ...     manager=wrapper.manager,
        ...     limit_resolver=wrapper._resolve_limits,
        ...     default_model="pegasus-1",
        ...     excluded_kwargs=["model"],  # TwelveLabs doesn't accept model
        ... )
    """
    # Auto-detect async if not specified
    if is_async is None:
        is_async = detect_async_client(client_class)

    # Introspect valid constructor kwargs
    valid_kwargs = get_valid_constructor_kwargs(client_class)

    config = GenericClientConfig(
        client_class=client_class,
        api_key_param=api_key_param,
        is_async=is_async,
        usage_extractor=usage_extractor,
        estimated_tokens=estimated_tokens,
        max_retries=max_retries,
        model_param=model_param,
        client_kwargs=client_kwargs,
        excluded_kwargs=frozenset(excluded_kwargs or []),
        valid_kwargs=valid_kwargs,
    )

    if is_async:
        return AsyncGenericRotatingClient(
            manager=manager,
            limit_resolver=limit_resolver,
            default_model=default_model,
            config=config,
        )
    else:
        return SyncGenericRotatingClient(
            manager=manager,
            limit_resolver=limit_resolver,
            default_model=default_model,
            config=config,
        )
