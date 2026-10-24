import asyncio
import atexit
import logging
import threading
import time
from threading import Event, Lock
from typing import Callable, Optional, Union

from ..config.constants import (
    CLEANUP_INTERVAL_SECONDS,
    DEFAULT_COOLDOWN_SECONDS,
    HISTORY_LOOKBACK_SECONDS,
    KEY_WAIT_MAX_SECONDS,
    KEY_WAIT_POLL_SECONDS,
)
from ..config.dataclasses import (
    GlobalStats,
    KeyDetailedStats,
    KeySummary,
    KeyUsage,
    ModelAggregatedStats,
    RateLimits,
    UsageSnapshot,
)
from ..config.enums import RateLimitStrategy
from ..config.log_config import default_logger
from ..core.exceptions import ConfigurationError
from ..core.utils import KeyEntry, get_key_suffix, normalize_key_entries
from ..usage.db_logic import UsageDatabase
from ..usage.usage_logger import AsyncUsageLogger


class RotatingKeyManager:
    """Manages API key rotation with rate limiting"""

    def __init__(
        self,
        api_keys: list[KeyEntry],
        provider_name: str,
        strategy: RateLimitStrategy,
        db: Optional[UsageDatabase] = None,
        logger: Optional[logging.Logger] = None,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        limit_resolver: Optional[Callable[[str, Optional[str]], RateLimits]] = None,
        api_key_param: str = "api_key",
        max_key_wait_seconds: float = KEY_WAIT_MAX_SECONDS,
        track_usage: bool = True,
    ):
        self.provider_name = provider_name
        self.max_key_wait_seconds = max_key_wait_seconds
        self.logger = logger or default_logger
        self.strategy = strategy
        self.cooldown_seconds = cooldown_seconds
        self.limit_resolver = limit_resolver
        self.api_key_param = api_key_param

        # Normalize key entries and create KeyUsage objects with params
        normalized = normalize_key_entries(api_keys, api_key_param)
        self.keys = [
            KeyUsage(api_key=primary, strategy=strategy, params=params)
            for primary, params in normalized
        ]
        self.current_index = 0
        self.lock = Lock()

        self.db = db
        # Without a database there is nothing to persist to, so tracking is
        # forced off and the manager runs purely in memory.
        self.track_usage = bool(track_usage) and db is not None
        # The logger exists whenever a db does, so `manager.track_usage = True`
        # at runtime resumes writes without rebuilding the writer thread.
        self.usage_logger = AsyncUsageLogger(db) if db is not None else None
        if self.track_usage:
            self._hydrate()

        self._stop_event = Event()
        self._start_cleanup()
        atexit.register(self.stop)

        self.logger.info("Initialized %d keys for provider %s.", len(self.keys), provider_name)

    def force_rotate_index(self) -> None:
        """
        Force the internal pointer to increment.
        Useful when a key hits a 429 despite local checks passing.
        """
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.keys)

    def _hydrate(self) -> None:
        """Load historical usage from database to restore state."""
        if self.db is None:
            return
        self.logger.debug("Loading history for provider %s.", self.provider_name)
        all_history = self.db.load_provider_history(self.provider_name, HISTORY_LOOKBACK_SECONDS)
        if not all_history:
            self.logger.info("No history found in DB for %s.", self.provider_name)
            return

        key_map = {get_key_suffix(k.api_key): k for k in self.keys}
        count = 0
        for row in all_history:
            suffix, model_id, ts, tokens = row

            if suffix in key_map:
                key_map[suffix].record_usage(model_id, tokens=tokens, timestamp=ts)
                count += 1
        self.logger.info("Hydrated %d records for %s.", count, self.provider_name)

    def _cleanup_loop(self) -> None:
        """Periodically clean deques to prevent memory bloat."""
        while not self._stop_event.is_set():
            try:
                with self.lock:
                    for key in self.keys:
                        for bucket in key.buckets.values():
                            bucket.clean()
                        if self.strategy == RateLimitStrategy.GLOBAL:
                            key.global_bucket.clean()
            except Exception as e:
                self.logger.error("Cleanup loop error: %s", e, exc_info=True)
            # Wait on the stop event, not the clock: stop() wakes this loop at
            # once instead of leaving join() to time out while the thread
            # sleeps through the whole interval.
            self._stop_event.wait(CLEANUP_INTERVAL_SECONDS)

    def _start_cleanup(self) -> None:
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the cleanup thread and flush logs. Returns promptly: the
        cleanup loop waits on the stop event, so the join does not wait out
        CLEANUP_INTERVAL_SECONDS."""
        self._stop_event.set()
        if self.usage_logger is not None:
            self.usage_logger.stop()  # Flush logs
        if self._thread.is_alive():
            self._thread.join(timeout=10)

    def get_key(
        self, model_id: str, default_limits: RateLimits, estimated_tokens: int = 1000
    ) -> Optional[KeyUsage]:
        """
        Get an available API key that can handle the request.

        Args:
            model_id: The model identifier for rate limit lookup
            default_limits: Default limits to use if no per-key override exists
            estimated_tokens: Estimated token usage for this request

        Returns:
            KeyUsage object if a key is available, None otherwise
        """
        with self.lock:
            for offset in range(len(self.keys)):
                idx = (self.current_index + offset) % len(self.keys)
                key: KeyUsage = self.keys[idx]

                if key.is_cooling_down(self.cooldown_seconds):
                    continue

                # Resolve limits for this specific key (supports per-key overrides)
                if self.limit_resolver:
                    key_suffix = get_key_suffix(key.api_key)
                    limits = self.limit_resolver(model_id, key_suffix)
                else:
                    limits = default_limits

                if key.can_use_model(model_id, limits, estimated_tokens):
                    key.reserve(model_id, estimated_tokens)
                    self.current_index = idx
                    return key
            return None

    def all_keys_dead(self) -> bool:
        """True when every key has been marked dead (401/402/403) - nothing to wait for."""
        return bool(self.keys) and all(k.dead for k in self.keys)

    def unavailable_message(self, model_id: str) -> str:
        dead = sum(1 for k in self.keys if k.dead)
        if self.all_keys_dead():
            return (
                f"No usable keys for {model_id}: all {len(self.keys)} keys are dead (401/402/403)"
            )
        return (
            f"No available keys for {model_id}: {dead}/{len(self.keys)} dead, the rest stayed "
            f"rate-limited or cooling down for {self.max_key_wait_seconds:.0f}s"
        )

    def acquire_key(
        self,
        model_id: str,
        default_limits: RateLimits,
        estimated_tokens: int = 1000,
        max_wait_seconds: Optional[float] = None,
        poll_seconds: float = KEY_WAIT_POLL_SECONDS,
    ) -> Optional[KeyUsage]:
        """
        get_key, but WAIT while every live key is inside a rate window or a
        cooldown instead of handing back None at once.

        Returns None only when every key is dead or max_wait elapses - so a
        caller with one production key at its per-minute cap rides the window
        out rather than aborting a long run.
        """
        max_wait = self.max_key_wait_seconds if max_wait_seconds is None else max_wait_seconds
        deadline = time.monotonic() + max_wait
        while True:
            key = self.get_key(model_id, default_limits, estimated_tokens)
            if key is not None:
                return key
            if self.all_keys_dead() or time.monotonic() >= deadline:
                return None
            time.sleep(poll_seconds)

    async def acquire_key_async(
        self,
        model_id: str,
        default_limits: RateLimits,
        estimated_tokens: int = 1000,
        max_wait_seconds: Optional[float] = None,
        poll_seconds: float = KEY_WAIT_POLL_SECONDS,
    ) -> Optional[KeyUsage]:
        """acquire_key for async callers (non-blocking sleep)."""
        max_wait = self.max_key_wait_seconds if max_wait_seconds is None else max_wait_seconds
        deadline = time.monotonic() + max_wait
        while True:
            key = self.get_key(model_id, default_limits, estimated_tokens)
            if key is not None:
                return key
            if self.all_keys_dead() or time.monotonic() >= deadline:
                return None
            await asyncio.sleep(poll_seconds)

    def get_specific_key(
        self, identifier: Union[int, str], model_id: str, estimated_tokens: int = 1000
    ) -> Optional[KeyUsage]:
        """
        Get a specific key by index or identifier.
        Note: This does NOT check rate limits implicitly to allow 'hard' retrieval,
        but it DOES reserve the estimated tokens to keep tracking accurate.
        """
        with self.lock:
            key, _ = self._find_key(identifier)
            if key:
                key.reserve(model_id, estimated_tokens)
                return key
            return None

    def record_usage(
        self,
        key_obj: KeyUsage,
        model_id: str,
        actual_tokens: int,
        estimated_tokens: int = 1000,
        track_usage: Optional[bool] = None,
    ) -> None:
        """
        Record usage for a specific API key.

        In-memory accounting (what drives rotation) always happens. The database
        write happens only when tracking is on - `track_usage` overrides the
        manager's own setting for this one call (None = use the manager's).
        """
        with self.lock:
            key_obj.commit(model_id, actual_tokens, estimated_tokens)

        should_log = self.track_usage if track_usage is None else bool(track_usage)
        if not should_log:
            return
        if self.usage_logger is None:
            raise ConfigurationError(
                "usage tracking was requested but no usage database is configured "
                f"for provider '{self.provider_name}'. Build the manager with a "
                "UsageDatabase, or record usage with track_usage=False."
            )
        self.usage_logger.log(self.provider_name, model_id, key_obj.api_key, actual_tokens)

    def prune_old_records(self, days_retention: int = 3) -> None:
        """Prune old rows from the usage database. No-op when there is no db."""
        if self.db is None:
            return
        self.db.prune_old_records(days_retention)

    # --- STATS HELPERS ---

    def _find_key(self, identifier: Union[int, str]) -> tuple[Optional[KeyUsage], int]:
        """Locate key by index (int) or suffix/full-key (str)."""
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.keys):
                return self.keys[identifier], identifier
        elif isinstance(identifier, str):
            for i, k in enumerate(self.keys):
                if k.api_key == identifier or k.api_key.endswith(identifier):
                    return k, i
        return None, -1

    def get_global_stats(self) -> GlobalStats:
        """Aggregates usage across all keys and models."""
        total = UsageSnapshot()
        keys_summary = []
        with self.lock:
            for i, key in enumerate(self.keys):
                snap = key.get_total_snapshot()
                total = total + snap
                suffix = get_key_suffix(key.api_key)
                keys_summary.append(KeySummary(index=i, suffix=suffix, snapshot=snap))
        return GlobalStats(total=total, keys=keys_summary)

    def get_key_stats(self, identifier: Union[int, str]) -> Optional[KeyDetailedStats]:
        """Stats for a specific key, including per-model breakdown."""
        with self.lock:
            key, idx = self._find_key(identifier)
            if not key:
                return None
            total_snap = key.get_total_snapshot()
            breakdown = {}
            for model, bucket in key.buckets.items():
                breakdown[model] = bucket.get_snapshot()
            suffix = get_key_suffix(key.api_key)
            return KeyDetailedStats(index=idx, suffix=suffix, total=total_snap, breakdown=breakdown)

    def get_model_stats(self, model_id: str) -> ModelAggregatedStats:
        """Aggregates stats for ONE model across ALL keys."""
        total = UsageSnapshot()
        contributing_keys = []
        with self.lock:
            for i, key in enumerate(self.keys):
                if model_id in key.buckets:
                    snap = key.buckets[model_id].get_snapshot()
                    total = total + snap
                    suffix = get_key_suffix(key.api_key)
                    contributing_keys.append(KeySummary(index=i, suffix=suffix, snapshot=snap))
        return ModelAggregatedStats(model_id=model_id, total=total, keys=contributing_keys)

    def get_granular_stats(
        self, identifier: Union[int, str], model_id: str
    ) -> Optional[KeySummary]:
        """Specific Key + Specific Model."""
        with self.lock:
            key, idx = self._find_key(identifier)
            if not key:
                return None
            suffix = get_key_suffix(key.api_key)
            snap = (
                key.buckets[model_id].get_snapshot() if model_id in key.buckets else UsageSnapshot()
            )
            return KeySummary(index=idx, suffix=suffix, snapshot=snap)
