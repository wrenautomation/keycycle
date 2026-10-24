"""
acquire_key: wait for a rate window / cooldown instead of failing at once.

One production key at its per-minute cap (Cohere, emails_gen pilot
2026-09-07) used to surface as "No available keys" after the first
window filled, aborting a 2,000-document run 20 calls in.
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from keycycle.config.dataclasses import RateLimits
    from keycycle.config.enums import RateLimitStrategy
    from keycycle.key_rotation.rotation_manager import RotatingKeyManager
except ImportError:
    from keycycle.keycycle.config.dataclasses import RateLimits
    from keycycle.keycycle.config.enums import RateLimitStrategy
    from keycycle.keycycle.key_rotation.rotation_manager import RotatingKeyManager


def _manager(keys, cooldown_seconds=1):
    db = MagicMock()
    db.load_provider_history.return_value = []
    return RotatingKeyManager(
        api_keys=keys,
        provider_name="test",
        strategy=RateLimitStrategy.PER_MODEL,
        db=db,
        cooldown_seconds=cooldown_seconds,
    )


class TestAcquireKey(unittest.TestCase):
    def setUp(self):
        self.limits = RateLimits(100, 6000, 10000, 1000000)

    def test_returns_immediately_when_a_key_is_free(self):
        manager = _manager(["sk-only-key-AAAAAAAA"])
        started = time.monotonic()
        key = manager.acquire_key("m", self.limits, estimated_tokens=10)
        self.assertIsNotNone(key)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_waits_out_a_cooldown_then_returns_the_key(self):
        manager = _manager(["sk-only-key-AAAAAAAA"], cooldown_seconds=1)
        manager.keys[0].trigger_cooldown()
        self.assertIsNone(manager.get_key("m", self.limits, 10))  # the old behavior

        started = time.monotonic()
        key = manager.acquire_key("m", self.limits, 10, max_wait_seconds=5, poll_seconds=0.1)
        self.assertIsNotNone(key)
        self.assertGreaterEqual(time.monotonic() - started, 0.5)

    def test_all_dead_returns_none_without_waiting(self):
        manager = _manager(["sk-a-AAAAAAAA", "sk-b-BBBBBBBB"])
        for k in manager.keys:
            k.mark_dead()
        started = time.monotonic()
        self.assertIsNone(manager.acquire_key("m", self.limits, 10, max_wait_seconds=5))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIn("all 2 keys are dead", manager.unavailable_message("m"))

    def test_gives_up_after_max_wait(self):
        manager = _manager(["sk-only-key-AAAAAAAA"], cooldown_seconds=60)
        manager.keys[0].trigger_cooldown()
        started = time.monotonic()
        self.assertIsNone(
            manager.acquire_key("m", self.limits, 10, max_wait_seconds=0.3, poll_seconds=0.1)
        )
        self.assertLess(time.monotonic() - started, 2)
        self.assertIn("rate-limited or cooling down", manager.unavailable_message("m"))


if __name__ == "__main__":
    unittest.main()
