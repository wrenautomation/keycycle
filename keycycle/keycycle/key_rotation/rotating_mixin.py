import asyncio
import functools
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Optional, Union

from ..config.constants import (
    TEMP_RATE_LIMIT_INITIAL_DELAY,
    TEMP_RATE_LIMIT_MAX_DELAY,
    TEMP_RATE_LIMIT_MAX_RETRIES,
    TEMP_RATE_LIMIT_MULTIPLIER,
)
from ..config.dataclasses import KeyUsage
from ..config.log_config import default_logger
from ..core.backoff import BackoffConfig, ExponentialBackoff
from ..core.utils import (
    get_key_suffix,
    is_dead_key_error,
    is_rate_limit_error,
    is_temporary_rate_limit_error,
)

if TYPE_CHECKING:
    from agno.models.response import ModelResponse


class _UsageResponse:
    """Minimal response holder for usage recording when agno is not available."""

    __slots__ = ("response_usage",)

    def __init__(self, usage=None):
        self.response_usage = usage


class RotatingCredentialsMixin:
    """
    Mixin that handles key rotation, 429 detection, and 30s cooldown triggers.
    """

    def __init__(
        self,
        *args,
        model_id: str,
        wrapper=None,
        rotating_wait=True,
        rotating_timeout=10.0,
        rotating_estimated_tokens=1000,
        rotating_max_retries=5,
        rotating_fixed_key_id: Union[int, str] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """
        Initializes rotation parameters and patches the model if necessary.

        Args:
            wrapper: The MultiProviderWrapper instance managing keys.
            rotating_wait: Whether to wait for a key if none are available.
            rotating_timeout: Max time to wait for a key.
            rotating_estimated_tokens: Default token estimate for rate limiting.
            rotating_max_retries: Number of retries on 429 errors.
        """
        self.logger = logger or default_logger
        self.wrapper = wrapper
        self.model_id = model_id
        self._rotating_wait = rotating_wait
        self._rotating_timeout = rotating_timeout
        self._estimated_tokens = rotating_estimated_tokens
        self._max_retries = rotating_max_retries
        self._fixed_key_id = rotating_fixed_key_id

        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        """
        Hook to automatically copy function signatures and docstrings
        from the parent class to this mixin's wrapper methods.
        """
        super().__init_subclass__(**kwargs)
        methods_to_sync = ["invoke", "ainvoke", "invoke_stream", "ainvoke_stream"]
        for method_name in methods_to_sync:
            if not hasattr(cls, method_name):
                continue
            target_method = None
            for parent in cls.mro():
                if parent is cls or parent is RotatingCredentialsMixin:
                    continue
                # Check if the method is defined in this specific parent's __dict__
                if hasattr(parent, method_name) and method_name in parent.__dict__:
                    target_method = getattr(parent, method_name)
                    break

            if target_method:
                mixin_implementation = getattr(cls, method_name)
                wrapped_method = functools.wraps(target_method)(mixin_implementation)

                if hasattr(target_method, "__signature__"):
                    wrapped_method.__signature__ = target_method.__signature__
                setattr(cls, method_name, wrapped_method)

    def _rotate_credentials(self) -> KeyUsage:
        key_usage: KeyUsage = self.wrapper.get_key_usage(
            model_id=self.model_id,
            estimated_tokens=self._estimated_tokens,
            wait=self._rotating_wait,
            timeout=self._rotating_timeout,
            key_id=self._fixed_key_id,
        )
        self.api_key = key_usage.api_key

        if hasattr(self, "client"):
            self.client = None
        if hasattr(self, "async_client"):
            self.async_client = None
        if hasattr(self, "gemini_client"):
            self.gemini_client = None

        return key_usage

    def _get_retry_limit(self) -> int:
        return min(self._max_retries, len(self.wrapper.manager.keys) - 1)

    def _record_usage(
        self,
        key_obj: KeyUsage,
        response: Optional["ModelResponse"],
        track_usage: Optional[bool] = None,
    ):
        """
        Extracts usage from the response and reports it to the manager.
        Falls back to estimated_tokens if response is None or usage is missing.
        """
        if not self.wrapper:
            return

        actual_tokens = 0

        if response and response.response_usage:
            actual_tokens = response.response_usage.total_tokens

        # Fallback: If no usage found (or streaming), use the estimate
        if actual_tokens == 0:
            actual_tokens = self._estimated_tokens

        self.wrapper.manager.record_usage(
            key_obj=key_obj,
            model_id=self.model_id,
            actual_tokens=actual_tokens,
            estimated_tokens=self._estimated_tokens,
            track_usage=track_usage,
        )

    def _create_temp_backoff(self) -> ExponentialBackoff:
        """Create a backoff instance for temporary rate limit retries."""
        return ExponentialBackoff(
            BackoffConfig(
                initial_interval=TEMP_RATE_LIMIT_INITIAL_DELAY,
                max_interval=TEMP_RATE_LIMIT_MAX_DELAY,
                multiplier=TEMP_RATE_LIMIT_MULTIPLIER,
            )
        )

    def invoke(self, *args, **kwargs) -> "ModelResponse":
        # Per-call tracking override - popped before the kwargs reach the
        # underlying model, which does not know this keyword.
        track_usage = kwargs.pop("track_usage", None)
        limit = self._get_retry_limit()

        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            temp_backoff = self._create_temp_backoff()

            for temp_attempt in range(TEMP_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    response = super().invoke(*args, **kwargs)
                    self._record_usage(key_usage, response, track_usage)
                    return response
                except Exception as e:
                    # Check for temporary rate limit first - retry with SAME key
                    if (
                        is_temporary_rate_limit_error(e)
                        and temp_attempt < TEMP_RATE_LIMIT_MAX_RETRIES
                    ):
                        delay = temp_backoff.get_next_interval()
                        self.logger.info(
                            "Temporary rate limit on key %s [%s]. Waiting %.1fs (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            delay,
                            temp_attempt + 1,
                            TEMP_RATE_LIMIT_MAX_RETRIES,
                        )
                        time.sleep(delay)
                        continue  # Retry with SAME key

                    # Dead key (401/403 revoked/denied, 402 credits exhausted) - key is dead
                    # for this process, rotate to next key
                    if is_dead_key_error(e) and attempt < limit:
                        self.logger.warning(
                            "Dead key (401/402/403) on key %s (Sync) [%s]. Marking key dead and rotating (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.mark_dead()
                        self.wrapper.manager.force_rotate_index()
                        break  # Break inner loop, continue outer with new key

                    # Hard rate limit - rotate key
                    if is_rate_limit_error(e) and attempt < limit:
                        self.logger.warning(
                            "429 Hit on key %s (Sync) [%s]. Rotating and retrying (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.trigger_cooldown()
                        self.wrapper.manager.force_rotate_index()
                        break  # Break inner loop, continue outer with new key
                    raise
            else:
                # Temp retries exhausted, move to next key
                if attempt < limit:
                    self.logger.warning(
                        "Temp rate limit retries exhausted for key %s. Rotating.",
                        get_key_suffix(self.api_key),
                    )
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise

    async def ainvoke(self, *args, **kwargs) -> "ModelResponse":
        # Per-call tracking override - popped before the kwargs reach the
        # underlying model, which does not know this keyword.
        track_usage = kwargs.pop("track_usage", None)
        limit = self._get_retry_limit()

        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            temp_backoff = self._create_temp_backoff()

            for temp_attempt in range(TEMP_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    response = await super().ainvoke(*args, **kwargs)
                    self._record_usage(key_usage, response, track_usage)
                    return response
                except Exception as e:
                    # Check for temporary rate limit first - retry with SAME key
                    if (
                        is_temporary_rate_limit_error(e)
                        and temp_attempt < TEMP_RATE_LIMIT_MAX_RETRIES
                    ):
                        delay = temp_backoff.get_next_interval()
                        self.logger.info(
                            "Temporary rate limit on key %s [%s]. Waiting %.1fs (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            delay,
                            temp_attempt + 1,
                            TEMP_RATE_LIMIT_MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue  # Retry with SAME key

                    # Dead key (401/403 revoked/denied, 402 credits exhausted) - key is dead
                    # for this process, rotate to next key
                    if is_dead_key_error(e) and attempt < limit:
                        self.logger.warning(
                            "Dead key (401/402/403) on key %s (Async) [%s]. Marking key dead and rotating (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.mark_dead()
                        self.wrapper.manager.force_rotate_index()
                        break  # Break inner loop, continue outer with new key

                    # Hard rate limit - rotate key
                    if is_rate_limit_error(e) and attempt < limit:
                        self.logger.warning(
                            "429 Hit on key %s (Async) [%s]. Rotating and retrying (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.trigger_cooldown()
                        self.wrapper.manager.force_rotate_index()
                        break  # Break inner loop, continue outer with new key
                    raise
            else:
                # Temp retries exhausted, move to next key
                if attempt < limit:
                    self.logger.warning(
                        "Temp rate limit retries exhausted for key %s. Rotating.",
                        get_key_suffix(self.api_key),
                    )
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise

    def invoke_stream(self, *args, **kwargs) -> Iterator["ModelResponse"]:
        # Per-call tracking override - popped before the kwargs reach the
        # underlying model, which does not know this keyword.
        track_usage = kwargs.pop("track_usage", None)
        limit = self._get_retry_limit()

        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            temp_backoff = self._create_temp_backoff()

            for temp_attempt in range(TEMP_RATE_LIMIT_MAX_RETRIES + 1):
                chunks_received = False
                try:
                    stream = super().invoke_stream(*args, **kwargs)
                    final_usage = None
                    for chunk in stream:
                        chunks_received = True
                        if chunk.response_usage:
                            final_usage = chunk.response_usage
                        yield chunk
                    if final_usage:
                        dummy_response = _UsageResponse(final_usage)
                        self._record_usage(key_usage, dummy_response, track_usage)
                    else:
                        # If the stream didn't return usage data, fallback to estimation
                        self._record_usage(key_usage, None, track_usage)

                    return
                except Exception as e:
                    # Only retry temp limits if NO chunks received yet
                    if (
                        is_temporary_rate_limit_error(e)
                        and temp_attempt < TEMP_RATE_LIMIT_MAX_RETRIES
                        and not chunks_received
                    ):
                        delay = temp_backoff.get_next_interval()
                        self.logger.info(
                            "Temporary rate limit on key %s (Stream) [%s]. Waiting %.1fs (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            delay,
                            temp_attempt + 1,
                            TEMP_RATE_LIMIT_MAX_RETRIES,
                        )
                        time.sleep(delay)
                        continue

                    if is_dead_key_error(e) and attempt < limit:
                        self.logger.warning(
                            "Dead key (401/402/403) on key %s (Sync Stream) [%s]. Marking key dead and rotating (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.mark_dead()
                        self.wrapper.manager.force_rotate_index()
                        break

                    if is_rate_limit_error(e) and attempt < limit:
                        self.logger.warning(
                            "429 Hit on key %s (Sync Stream) [%s]. Rotating and retrying (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.trigger_cooldown()
                        self.wrapper.manager.force_rotate_index()
                        break
                    raise
            else:
                if attempt < limit:
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator["ModelResponse"]:
        # Per-call tracking override - popped before the kwargs reach the
        # underlying model, which does not know this keyword.
        track_usage = kwargs.pop("track_usage", None)
        limit = self._get_retry_limit()

        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            temp_backoff = self._create_temp_backoff()

            for temp_attempt in range(TEMP_RATE_LIMIT_MAX_RETRIES + 1):
                chunks_received = False
                try:
                    stream = super().ainvoke_stream(*args, **kwargs)

                    final_usage = None
                    async for chunk in stream:
                        chunks_received = True
                        if chunk.response_usage:
                            final_usage = chunk.response_usage
                        yield chunk

                    # Stream completed successfully. Record usage.
                    if final_usage:
                        dummy_response = _UsageResponse(final_usage)
                        self._record_usage(key_usage, dummy_response, track_usage)
                    else:
                        self._record_usage(key_usage, None, track_usage)

                    return
                except Exception as e:
                    # Only retry temp limits if NO chunks received yet
                    if (
                        is_temporary_rate_limit_error(e)
                        and temp_attempt < TEMP_RATE_LIMIT_MAX_RETRIES
                        and not chunks_received
                    ):
                        delay = temp_backoff.get_next_interval()
                        self.logger.info(
                            "Temporary rate limit on key %s (Async Stream) [%s]. Waiting %.1fs (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            delay,
                            temp_attempt + 1,
                            TEMP_RATE_LIMIT_MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if is_dead_key_error(e) and attempt < limit:
                        self.logger.warning(
                            "Dead key (401/402/403) on key %s (Async Stream) [%s]. Marking key dead and rotating (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.mark_dead()
                        self.wrapper.manager.force_rotate_index()
                        break

                    if is_rate_limit_error(e) and attempt < limit:
                        self.logger.warning(
                            "429 Hit on key %s (Async Stream) [%s]. Rotating and retrying (%d/%d).",
                            get_key_suffix(self.api_key),
                            self.model_id,
                            attempt + 1,
                            limit,
                        )
                        key_usage.trigger_cooldown()
                        self.wrapper.manager.force_rotate_index()
                        break
                    raise
            else:
                if attempt < limit:
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise
