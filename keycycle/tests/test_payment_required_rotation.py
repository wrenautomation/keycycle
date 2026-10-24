"""
Unit tests for HTTP 402 (payment required) key rotation.

Mirrors the style of test_rate_limit_logic.py (detection-level tests) and
test_custom_key_limits.py (RotatingKeyManager-level tests), extended to cover
the new 402/payment_required rotation trigger: a key whose free-tier
quota/credits are exhausted should be marked dead and excluded from rotation
for the rest of the process, instead of aborting the whole run.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from keycycle.adapters.openai_adapter import RotatingOpenAIClient
    from keycycle.config.dataclasses import KeyUsage, RateLimits
    from keycycle.config.enums import RateLimitStrategy
    from keycycle.core.utils import is_payment_required_error, is_rate_limit_error
    from keycycle.key_rotation.rotation_manager import RotatingKeyManager
except ImportError:
    # Fallback for different path structures or if run directly
    from keycycle.keycycle.adapters.openai_adapter import RotatingOpenAIClient
    from keycycle.keycycle.config.dataclasses import KeyUsage, RateLimits
    from keycycle.keycycle.config.enums import RateLimitStrategy
    from keycycle.keycycle.core.utils import is_payment_required_error, is_rate_limit_error
    from keycycle.keycycle.key_rotation.rotation_manager import RotatingKeyManager


class MockAPIError(Exception):
    """Mirrors the shape of openai's APIStatusError: message + status_code + body."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        # Some libraries put body in response
        self.response = body


class TestPaymentRequiredDetection(unittest.TestCase):
    """Test cases for detecting HTTP 402 payment_required errors."""

    def test_status_code_402_detected(self):
        """Test detection via status_code attribute (e.g. openai's APIStatusError)."""
        e = MockAPIError("Payment Required", status_code=402)
        self.assertTrue(is_payment_required_error(e), "Failed status_code=402 check")

    def test_payment_required_keyword_detected(self):
        """Test detection via the literal 'payment_required' error type/message."""
        error_message = (
            "Error code: 402 - {'error': {'message': 'You have exceeded your "
            "free trial credits', 'type': 'payment_required', 'code': 402}}"
        )
        e = Exception(error_message)
        self.assertTrue(is_payment_required_error(e), "Failed to detect payment_required keyword")

    def test_payment_required_words_detected(self):
        """Test detection via the human-readable 'Payment Required' phrase."""
        e = Exception("402 Payment Required: insufficient credits")
        self.assertTrue(is_payment_required_error(e), "Failed to detect 'Payment Required' phrase")

    def test_body_detection(self):
        """Test detection via body content when the top-level message is generic."""
        e = MockAPIError(
            "Provider returned error", body={"message": "Payment Required", "code": 402}
        )
        self.assertTrue(is_payment_required_error(e), "Failed body check")

    def test_non_payment_error_returns_false(self):
        """Unrelated errors should not be misdetected as payment_required."""
        e = Exception("Connection timeout")
        self.assertFalse(is_payment_required_error(e), "Non-402 error incorrectly flagged")

    def test_429_is_not_payment_required(self):
        """A 429 rate limit must not be misclassified as a 402."""
        e = MockAPIError("Rate limit exceeded", status_code=429)
        self.assertFalse(
            is_payment_required_error(e), "429 incorrectly flagged as payment_required"
        )

    def test_402_is_not_rate_limit(self):
        """A 402 must not be misclassified as a 429 rate limit (no overlap)."""
        e = MockAPIError("Payment Required", status_code=402)
        self.assertFalse(is_rate_limit_error(e), "402 incorrectly flagged as a rate limit error")


class TestKeyUsageMarkDead(unittest.TestCase):
    """Test the KeyUsage.mark_dead() / is_cooling_down() dead-key mechanism."""

    def setUp(self):
        self.key = KeyUsage(
            api_key="sk-test-dead-key-xxxxxxxx", strategy=RateLimitStrategy.PER_MODEL
        )

    def test_fresh_key_is_not_cooling_down(self):
        self.assertFalse(self.key.is_cooling_down())

    def test_mark_dead_sets_dead_flag(self):
        self.key.mark_dead()
        self.assertTrue(self.key.dead)

    def test_dead_key_reports_cooling_down(self):
        """A dead key must be excluded from rotation via the existing cooldown check."""
        self.key.mark_dead()
        self.assertTrue(self.key.is_cooling_down())

    def test_dead_key_never_expires(self):
        """Unlike trigger_cooldown(), mark_dead() must not clear after any window."""
        self.key.mark_dead()
        # Even a cooldown window of 0 seconds shouldn't let a dead key back in
        self.assertTrue(self.key.is_cooling_down(cooldown_seconds=0))

    def test_trigger_cooldown_does_not_mark_dead(self):
        """A regular 429 cooldown must not permanently kill the key."""
        self.key.trigger_cooldown()
        self.assertFalse(self.key.dead)


class TestRotatingKeyManagerSkipsDeadKeys(unittest.TestCase):
    """Test that RotatingKeyManager.get_key() skips dead keys during rotation."""

    def setUp(self):
        self.api_keys = [
            "sk-test-key-one-AAAAAAAA",
            "sk-test-key-two-BBBBBBBB",
            "sk-test-key-three-CCCCCCCC",
        ]
        self.mock_db = MagicMock()
        self.mock_db.load_provider_history.return_value = []
        self.limits = RateLimits(100, 6000, 10000, 1000000)

    def test_dead_key_is_skipped(self):
        manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=self.mock_db,
        )
        manager.keys[0].mark_dead()

        key = manager.get_key("test-model", self.limits, estimated_tokens=100)

        self.assertIsNotNone(key)
        self.assertTrue(key.api_key.endswith("BBBBBBBB"))

    def test_rotation_finds_the_only_live_key_out_of_many(self):
        """Simulates the real scenario: several dead free-tier keys, one live one."""
        manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=self.mock_db,
        )
        manager.keys[0].mark_dead()
        manager.keys[1].mark_dead()

        key = manager.get_key("test-model", self.limits, estimated_tokens=100)

        self.assertIsNotNone(key)
        self.assertTrue(key.api_key.endswith("CCCCCCCC"))

    def test_all_keys_dead_returns_none(self):
        """If every key has 402'd, get_key must return None (no infinite loop)."""
        manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=self.mock_db,
        )
        for key in manager.keys:
            key.mark_dead()

        key = manager.get_key("test-model", self.limits, estimated_tokens=100)

        self.assertIsNone(key)


class TestOpenAIAdapter402Rotation(unittest.TestCase):
    """
    End-to-end test of RotatingOpenAIClient._execute rotating past 402s.

    Mirrors the real-world bug report: several free-tier keys have exhausted
    their quota (402 payment_required) but other keys still work. Rotation
    should skip the dead keys instead of aborting the whole request, and
    still surface a clear error (no infinite loop) if every key is dead.
    """

    def setUp(self):
        self.api_keys = [
            "sk-key-one-AAAAAAAA",
            "sk-key-two-BBBBBBBB",
            "sk-key-three-CCCCCCCC",
        ]
        self.mock_db = MagicMock()
        self.mock_db.load_provider_history.return_value = []
        self.manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=self.mock_db,
        )
        self.limits = RateLimits(100, 6000, 10000, 1000000)

        self.client = RotatingOpenAIClient(
            manager=self.manager,
            limit_resolver=lambda model_id, key_suffix: self.limits,
            default_model="test-model",
            max_retries=5,
            provider="cerebras",
        )

    def _fake_client_factory(self, dead_key_suffixes):
        """Build fake OpenAI-shaped clients whose create() 402s for dead keys."""

        def factory(api_key):
            fake = MagicMock()
            suffix = api_key[-8:]
            if suffix in dead_key_suffixes:
                fake.chat.completions.create.side_effect = MockAPIError(
                    "Payment Required", status_code=402
                )
            else:
                response = MagicMock()
                response.usage.total_tokens = 42
                fake.chat.completions.create.return_value = response
            return fake

        return factory

    def test_rotates_past_dead_keys_to_a_live_one(self):
        """Two keys 402, the third succeeds - the request should still succeed."""
        with (
            patch.object(
                self.client,
                "_get_fresh_client",
                side_effect=self._fake_client_factory({"AAAAAAAA", "BBBBBBBB"}),
            ),
            patch("keycycle.adapters.openai_adapter.time.sleep"),
        ):
            result = self.client.chat.completions.create(model="test-model")

        self.assertEqual(result.usage.total_tokens, 42)
        # The two exhausted keys should be marked dead for the rest of the process
        dead_suffixes = {k.api_key[-8:] for k in self.manager.keys if k.dead}
        self.assertEqual(dead_suffixes, {"AAAAAAAA", "BBBBBBBB"})

    def test_all_keys_dead_raises_clearly_without_hanging(self):
        """If every key 402s, the error must surface clearly - no infinite loop."""
        with (
            patch.object(
                self.client,
                "_get_fresh_client",
                side_effect=self._fake_client_factory({"AAAAAAAA", "BBBBBBBB", "CCCCCCCC"}),
            ),
            patch("keycycle.adapters.openai_adapter.time.sleep"),
        ):
            with self.assertRaises(Exception):
                self.client.chat.completions.create(model="test-model")

        # Every key should have been marked dead - none silently retried forever
        self.assertTrue(all(k.dead for k in self.manager.keys))


if __name__ == "__main__":
    unittest.main()
