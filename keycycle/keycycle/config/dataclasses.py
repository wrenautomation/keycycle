import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .constants import (
    DEFAULT_COOLDOWN_SECONDS,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
)
from .enums import RateLimitStrategy

# --- CONFIGURATION DATA ---


@dataclass
class RateLimits:
    """Rate limits for a provider"""

    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None


# Type alias for per-key rate limit overrides
# Can be a single RateLimits (applies to all models) or a dict mapping model_id -> RateLimits
KeyLimitOverride = Union[RateLimits, dict[str, RateLimits]]


@dataclass
class UsageSnapshot:
    """Standardized view of usage counters"""

    rpm: int = 0
    rph: int = 0
    rpd: int = 0
    tpm: int = 0
    tph: int = 0
    tpd: int = 0
    total_requests: int = 0
    total_tokens: int = 0

    def __add__(self, other):
        """Allow summing snapshots for aggregation"""
        if not isinstance(other, UsageSnapshot):
            return NotImplemented
        return UsageSnapshot(
            self.rpm + other.rpm,
            self.rph + other.rph,
            self.rpd + other.rpd,
            self.tpm + other.tpm,
            self.tph + other.tph,
            self.tpd + other.tpd,
            self.total_requests + other.total_requests,
            self.total_tokens + other.total_tokens,
        )


# --- STATS DATA TRANSFER OBJECTS (DTOs) ---


@dataclass
class KeySummary:
    index: int
    suffix: str
    snapshot: UsageSnapshot


@dataclass
class GlobalStats:
    total: UsageSnapshot
    keys: list[KeySummary]


@dataclass
class KeyDetailedStats:
    index: int
    suffix: str
    total: UsageSnapshot
    breakdown: dict[str, UsageSnapshot]


@dataclass
class ModelAggregatedStats:
    model_id: str
    total: UsageSnapshot
    keys: list[KeySummary]


# --- USAGE TRACKING ---
@dataclass
class UsageBucket:
    """Tracks counters for a SINGLE model context"""

    requests_minute: deque[float] = field(default_factory=deque)
    requests_hour: deque[float] = field(default_factory=deque)
    requests_day: deque[float] = field(default_factory=deque)

    tokens_minute: deque[tuple[float, int]] = field(default_factory=deque)
    tokens_hour: deque[tuple[float, int]] = field(default_factory=deque)
    tokens_day: deque[tuple[float, int]] = field(default_factory=deque)

    total_requests: int = 0
    total_tokens: int = 0

    pending_tokens: int = 0

    def clean(self) -> None:
        """Clean old entries based on current time"""
        now = time.time()
        cutoffs = (now - SECONDS_PER_MINUTE, now - SECONDS_PER_HOUR, now - SECONDS_PER_DAY)

        for d, cut in zip([self.requests_minute, self.requests_hour, self.requests_day], cutoffs):
            while d and d[0] <= cut:
                d.popleft()

        for d, cut in zip([self.tokens_minute, self.tokens_hour, self.tokens_day], cutoffs):
            while d and d[0][0] <= cut:
                d.popleft()

    def add(self, tokens: int, timestamp: float):
        self.requests_minute.append(timestamp)
        self.requests_hour.append(timestamp)
        self.requests_day.append(timestamp)
        self.total_requests += 1

        if tokens > 0:
            self.tokens_minute.append((timestamp, tokens))
            self.tokens_hour.append((timestamp, tokens))
            self.tokens_day.append((timestamp, tokens))
            self.total_tokens += tokens

    def check_limits(self, limits: RateLimits, estimated_tokens: int) -> bool:
        self.clean()
        if len(self.requests_minute) >= limits.requests_per_minute:
            return False
        if len(self.requests_hour) >= limits.requests_per_hour:
            return False
        if len(self.requests_day) >= limits.requests_per_day:
            return False

        current_tpm = sum(t[1] for t in self.tokens_minute) + self.pending_tokens
        current_tph = sum(t[1] for t in self.tokens_hour) + self.pending_tokens
        current_tpd = sum(t[1] for t in self.tokens_day) + self.pending_tokens

        if limits.tokens_per_minute and (current_tpm + estimated_tokens > limits.tokens_per_minute):
            return False
        if limits.tokens_per_hour and (current_tph + estimated_tokens > limits.tokens_per_hour):
            return False
        if limits.tokens_per_day and (current_tpd + estimated_tokens > limits.tokens_per_day):
            return False

        return True

    def reserve(self, tokens: int):
        """Lock in estimated tokens"""
        self.pending_tokens += tokens

    def commit(self, actual_tokens: int, reserved_tokens: int, timestamp: float):
        """Remove reservation and add actual usage"""
        self.pending_tokens -= reserved_tokens
        if self.pending_tokens < 0:
            import logging

            logging.getLogger(__name__).warning(
                "Pending tokens went negative (%d), clamping to 0", self.pending_tokens
            )
            self.pending_tokens = 0
        self.add(actual_tokens, timestamp)

    def get_snapshot(self) -> UsageSnapshot:
        """Return current counts as a clean snapshot"""
        self.clean()
        return UsageSnapshot(
            rpm=len(self.requests_minute),
            rph=len(self.requests_hour),
            rpd=len(self.requests_day),
            tpm=sum(t[1] for t in self.tokens_minute),
            tph=sum(t[1] for t in self.tokens_hour),
            tpd=sum(t[1] for t in self.tokens_day),
            total_requests=self.total_requests,
            total_tokens=self.total_tokens,
        )


@dataclass
class KeyUsage:
    """Represents an API Key and holds multiple UsageBuckets (one per model)"""

    api_key: str
    strategy: RateLimitStrategy
    params: dict[str, Any] = field(default_factory=dict)
    buckets: dict[str, UsageBucket] = field(default_factory=lambda: defaultdict(UsageBucket))
    global_bucket: UsageBucket = field(default_factory=UsageBucket)
    last_429: float = 0.0
    dead: bool = False

    def get_client_params(self) -> dict[str, Any]:
        """Returns all params for client instantiation."""
        if self.params:
            return dict(self.params)
        return {"api_key": self.api_key}

    def record_usage(self, model_id: str, tokens: int, timestamp: float = None):
        ts = timestamp if timestamp else time.time()
        self.buckets[model_id].add(tokens, ts)
        if self.strategy == RateLimitStrategy.GLOBAL:
            self.global_bucket.add(tokens, ts)

    def can_use_model(
        self, model_id: str, limits: RateLimits, estimated_tokens: int = 1000
    ) -> bool:
        """Check limits based on the provider's strategy"""
        if self.strategy == RateLimitStrategy.GLOBAL:
            return self.global_bucket.check_limits(limits, estimated_tokens)
        else:  # Per-Model Limits
            return self.buckets[model_id].check_limits(limits, estimated_tokens)

    def get_total_snapshot(self) -> UsageSnapshot:
        if self.strategy == RateLimitStrategy.GLOBAL:
            return self.global_bucket.get_snapshot()
        total = UsageSnapshot()
        for b in self.buckets.values():
            total = total + b.get_snapshot()
        return total

    def reserve(self, model_id: str, tokens: int):
        self.buckets[model_id].reserve(tokens)
        if self.strategy == RateLimitStrategy.GLOBAL:
            self.global_bucket.reserve(tokens)

    def commit(
        self, model_id: str, actual_tokens: int, reserved_tokens: int, timestamp: float = None
    ):
        ts = timestamp if timestamp else time.time()
        self.buckets[model_id].commit(actual_tokens, reserved_tokens, ts)
        if self.strategy == RateLimitStrategy.GLOBAL:
            self.global_bucket.commit(actual_tokens, reserved_tokens, ts)

    def is_cooling_down(self, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS) -> bool:
        """Returns True if the key is still in its cooldown penalty period, or dead."""
        if self.dead:
            return True
        if self.last_429 == 0:
            return False
        return (time.time() - self.last_429) < cooldown_seconds

    def trigger_cooldown(self):
        """Mark this key as rate-limited."""
        self.last_429 = time.time()

    def mark_dead(self):
        """
        Permanently exclude this key from rotation for the lifetime of this process.

        Used for errors a short cooldown can't fix - e.g. HTTP 402 payment_required
        when a free-trial key's credits/quota are exhausted. Unlike trigger_cooldown(),
        this never expires; the key stays dead until the process restarts.
        """
        self.dead = True
