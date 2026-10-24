import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any, Callable, Optional

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

try:
    from openai import AsyncOpenAI, OpenAI, RateLimitError

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

    # Generic placeholders
    class OpenAI:
        pass

    class AsyncOpenAI:
        pass

    RateLimitError = Exception

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "cerebras": "https://api.cerebras.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cohere": "https://api.cohere.ai/compatibility/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
}


class BaseRotatingClient:
    def __init__(
        self,
        manager: RotatingKeyManager,
        limit_resolver: Callable[[str, Optional[str]], RateLimits],
        default_model: str,
        estimated_tokens: int = 1000,
        max_retries: int = 5,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        client_kwargs: dict = None,
    ):
        """
        Initialize the rotating client.

        Args:
            manager: The key rotation manager
            limit_resolver: Function to resolve rate limits for a model
            default_model: Default model to use
            estimated_tokens: Estimated tokens per request
            max_retries: Maximum number of retries on rate limit
            base_url: Base URL for the API (takes precedence over provider)
            provider: Provider name (openai, openrouter, gemini, cerebras, groq)
            client_kwargs: Additional kwargs to pass to OpenAI client
        """

        if not HAS_OPENAI:
            raise ImportError(
                "The 'openai' library is required. Install with `pip install openai`."
            )

        self.manager = manager
        self.limit_resolver = limit_resolver
        self.default_model = default_model
        self.estimated_tokens = estimated_tokens
        self.max_retries = max_retries
        self.client_kwargs = client_kwargs or {}

        if base_url:
            self.base_url = base_url
        elif provider:
            provider_lower = provider.lower()
            if provider_lower not in PROVIDER_BASE_URLS:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Valid providers: {', '.join(PROVIDER_BASE_URLS.keys())}"
                )
            self.base_url = PROVIDER_BASE_URLS[provider_lower]
        else:
            self.base_url = None  # Will use OpenAI's default

        if self.base_url:
            self.client_kwargs["base_url"] = self.base_url

    def _record_usage(
        self,
        key_usage: KeyUsage,
        model_id: str,
        actual_tokens: int,
        track_usage: Optional[bool] = None,
    ) -> None:
        self.manager.record_usage(
            key_obj=key_usage,
            model_id=model_id,
            actual_tokens=actual_tokens,
            estimated_tokens=self.estimated_tokens,
            track_usage=track_usage,
        )

    def _extract_usage(self, response: Any) -> int:
        try:
            if hasattr(response, "usage") and response.usage:
                return response.usage.total_tokens
        except (AttributeError, TypeError):
            pass
        return 0


# --- SYNC IMPLEMENTATION ---


class RotatingOpenAIClient(BaseRotatingClient):
    def _get_fresh_client(self, api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key, **self.client_kwargs)

    def __getattr__(self, name):
        return SyncProxyHelper(self, [name])

    def _execute(self, path: list[str], args, kwargs):
        # Per-call tracking override - popped before the kwargs reach the real
        # OpenAI client, which rejects unknown keyword arguments.
        track_usage = kwargs.pop("track_usage", None)
        model_id = kwargs.get("model", self.default_model)
        if "model" not in kwargs:
            kwargs["model"] = model_id
        limits = self.limit_resolver(model_id, None)

        if kwargs.get("stream", False) and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        for attempt in range(self.max_retries + 1):
            key_usage = self.manager.acquire_key(model_id, limits, self.estimated_tokens)
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
                    real_client = self._get_fresh_client(key_usage.api_key)

                    target = real_client
                    for p in path:
                        target = getattr(target, p)

                    result = target(*args, **kwargs)

                    if kwargs.get("stream", False):
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
                    if is_dead_key_error(e) and attempt < self.max_retries:
                        logger.warning(
                            "Dead key (401/402/403) hit for %s on key ...%s. Marking key dead and rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        key_usage.mark_dead()
                        self.manager.force_rotate_index()
                        time.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    # Hard rate limit - rotate to next key
                    if is_rate_limit_error(e) and attempt < self.max_retries:
                        logger.warning(
                            "429/RateLimit hit for %s on key ...%s. Rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        key_usage.trigger_cooldown()
                        self.manager.force_rotate_index()
                        time.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    self._record_usage(key_usage, model_id, 0, track_usage)
                    raise
            else:
                # Inner loop exhausted without success - continue to next key
                if attempt < self.max_retries:
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
    ):
        final_tokens = 0
        try:
            for chunk in generator:
                if hasattr(chunk, "usage") and chunk.usage:
                    final_tokens = chunk.usage.total_tokens
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


class SyncProxyHelper:
    def __init__(self, client: RotatingOpenAIClient, path: list[str]):
        self.client = client
        self.path = path

    def __getattr__(self, name):
        return SyncProxyHelper(self.client, self.path + [name])

    def __call__(self, *args, **kwargs):
        return self.client._execute(self.path, args, kwargs)


# --- ASYNC IMPLEMENTATION ---


class RotatingAsyncOpenAIClient(BaseRotatingClient):
    def _get_fresh_client(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key, **self.client_kwargs)

    def __getattr__(self, name):
        return AsyncProxyHelper(self, [name])

    async def _execute(self, path: list[str], args, kwargs: dict):
        # Per-call tracking override - popped before the kwargs reach the real
        # OpenAI client, which rejects unknown keyword arguments.
        track_usage = kwargs.pop("track_usage", None)
        model_id = kwargs.get("model", self.default_model)
        if "model" not in kwargs:
            kwargs["model"] = model_id
        limits = self.limit_resolver(model_id, None)

        if kwargs.get("stream", False) and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        for attempt in range(self.max_retries + 1):
            key_usage = await self.manager.acquire_key_async(
                model_id, limits, self.estimated_tokens
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
                    real_client = self._get_fresh_client(key_usage.api_key)

                    target = real_client
                    for p in path:
                        target = getattr(target, p)

                    result = await target(*args, **kwargs)

                    if kwargs.get("stream", False):
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
                    if is_dead_key_error(e) and attempt < self.max_retries:
                        logger.warning(
                            "Dead key (401/402/403) hit for %s on key ...%s. Marking key dead and rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        key_usage.mark_dead()
                        self.manager.force_rotate_index()
                        await asyncio.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    # Hard rate limit - rotate to next key
                    if is_rate_limit_error(e) and attempt < self.max_retries:
                        logger.warning(
                            "429/RateLimit hit for %s on key ...%s. Rotating. (Attempt %d/%d)",
                            model_id,
                            get_key_suffix(key_usage.api_key),
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        key_usage.trigger_cooldown()
                        self.manager.force_rotate_index()
                        await asyncio.sleep(KEY_ROTATION_DELAY_SECONDS)
                        break  # Break inner loop, continue outer loop with new key

                    self._record_usage(key_usage, model_id, 0, track_usage)
                    raise
            else:
                # Inner loop exhausted without success - continue to next key
                if attempt < self.max_retries:
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
    ):
        final_tokens = 0
        try:
            async for chunk in generator:
                if hasattr(chunk, "usage") and chunk.usage:
                    final_tokens = chunk.usage.total_tokens
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


class AsyncProxyHelper:
    def __init__(self, client: RotatingAsyncOpenAIClient, path: list[str]):
        self.client = client
        self.path = path

    def __getattr__(self, name):
        return AsyncProxyHelper(self.client, self.path + [name])

    async def __call__(self, *args, **kwargs):
        return await self.client._execute(self.path, args, kwargs)
