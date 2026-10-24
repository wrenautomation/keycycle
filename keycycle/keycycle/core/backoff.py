"""Exponential backoff utilities for key rotation."""

import random
import time
from dataclasses import dataclass
from typing import Optional

from ..config.constants import DEFAULT_POLL_INTERVAL, MAX_POLL_INTERVAL, MIN_POLL_INTERVAL


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""

    initial_interval: float = DEFAULT_POLL_INTERVAL
    max_interval: float = MAX_POLL_INTERVAL
    multiplier: float = 2.0
    jitter: float = 0.1  # 10% jitter


class ExponentialBackoff:
    """
    Implements exponential backoff with jitter for key polling.

    Example:
        backoff = ExponentialBackoff()
        for attempt in range(max_attempts):
            key = try_get_key()
            if key:
                return key
            backoff.wait()
    """

    def __init__(self, config: Optional[BackoffConfig] = None):
        self.config = config or BackoffConfig()
        self._attempt = 0
        self._current_interval = self.config.initial_interval

    def reset(self) -> None:
        """Reset backoff state to initial values."""
        self._attempt = 0
        self._current_interval = self.config.initial_interval

    def get_next_interval(self) -> float:
        """
        Get the next backoff interval without waiting.

        Returns:
            The calculated interval with jitter applied
        """
        # Calculate base interval
        interval = min(self._current_interval, self.config.max_interval)

        # Apply jitter (+-jitter%)
        jitter_range = interval * self.config.jitter
        interval += random.uniform(-jitter_range, jitter_range)

        # Ensure minimum
        interval = max(interval, MIN_POLL_INTERVAL)

        # Update for next call
        self._current_interval *= self.config.multiplier
        self._attempt += 1

        return interval

    def wait(self) -> float:
        """
        Wait for the next backoff interval.

        Returns:
            The actual time waited
        """
        interval = self.get_next_interval()
        time.sleep(interval)
        return interval

    @property
    def attempt(self) -> int:
        """Current attempt number (0-indexed)."""
        return self._attempt
