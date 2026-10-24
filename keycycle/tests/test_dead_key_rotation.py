"""
Unit tests for dead-key (HTTP 401/402/403) rotation.

Extends test_payment_required_rotation.py: 401/403 mean the key itself is
unusable (revoked, invalid, or its project denied — Gemini answers
"Your project has been denied access" with a 403), exactly like a 402.
Such a key must be marked dead and rotated off, never allowed to abort a
whole run while dozens of healthy keys sit idle (emails_gen pilot,
2026-09-07).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from keycycle.adapters.openai_adapter import RotatingOpenAIClient
    from keycycle.config.dataclasses import RateLimits
    from keycycle.config.enums import RateLimitStrategy
    from keycycle.core.utils import is_dead_key_error
    from keycycle.key_rotation.rotation_manager import RotatingKeyManager
except ImportError:
    from keycycle.keycycle.adapters.openai_adapter import RotatingOpenAIClient
    from keycycle.keycycle.config.dataclasses import RateLimits
    from keycycle.keycycle.config.enums import RateLimitStrategy
    from keycycle.keycycle.core.utils import is_dead_key_error
    from keycycle.keycycle.key_rotation.rotation_manager import RotatingKeyManager


class MockAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = body


class TestDeadKeyDetection(unittest.TestCase):
    def test_403_project_denied_is_dead(self):
        e = MockAPIError(
            "Error code: 403 - [{'error': {'code': 403, 'message': 'Your project has "
            "been denied access. Please contact support.', 'status': 'PERMISSION_DENIED'}}]",
            status_code=403,
        )
        self.assertTrue(is_dead_key_error(e))

    def test_401_invalid_key_is_dead(self):
        self.assertTrue(is_dead_key_error(MockAPIError("Unauthorized", status_code=401)))

    def test_402_still_dead(self):
        self.assertTrue(is_dead_key_error(MockAPIError("Payment Required", status_code=402)))

    def test_429_never_dead_even_with_scary_message(self):
        # A rate limit whose body mentions "forbidden" is still just a rate limit.
        e = MockAPIError("429 rate limit: forbidden to exceed quota", status_code=429)
        self.assertFalse(is_dead_key_error(e))

    def test_string_fallback_without_status_code(self):
        self.assertTrue(is_dead_key_error(Exception("Error code: 403 - PERMISSION_DENIED")))
        self.assertFalse(is_dead_key_error(Exception("Something unrelated exploded")))


class TestDeadKeyRotation(unittest.TestCase):
    """A 403 on key A must mark A dead and let the request succeed on key B
    (same fixture shape as TestOpenAIAdapter402Rotation)."""

    def setUp(self):
        self.api_keys = ["sk-key-one-AAAAAAAA", "sk-key-two-BBBBBBBB"]
        mock_db = MagicMock()
        mock_db.load_provider_history.return_value = []
        self.manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=mock_db,
        )
        limits = RateLimits(100, 6000, 10000, 1000000)
        self.client = RotatingOpenAIClient(
            manager=self.manager,
            limit_resolver=lambda model_id, key_suffix: limits,
            default_model="test-model",
            max_retries=5,
            provider="gemini",
        )

    def _factory(self, denied_suffixes):
        def factory(api_key):
            fake = MagicMock()
            if api_key[-8:] in denied_suffixes:
                fake.chat.completions.create.side_effect = MockAPIError(
                    "Your project has been denied access. Please contact support.",
                    status_code=403,
                )
            else:
                response = MagicMock()
                response.usage.total_tokens = 7
                fake.chat.completions.create.return_value = response
            return fake

        return factory

    def test_403_marks_key_dead_and_rotates(self):
        with (
            patch.object(self.client, "_get_fresh_client", side_effect=self._factory({"AAAAAAAA"})),
            patch("keycycle.adapters.openai_adapter.time.sleep"),
        ):
            result = self.client.chat.completions.create(model="test-model")

        self.assertEqual(result.usage.total_tokens, 7)
        dead = {k.api_key[-8:] for k in self.manager.keys if k.dead}
        self.assertEqual(dead, {"AAAAAAAA"})

    def test_all_403_raises_without_hanging(self):
        with (
            patch.object(
                self.client,
                "_get_fresh_client",
                side_effect=self._factory({"AAAAAAAA", "BBBBBBBB"}),
            ),
            patch("keycycle.adapters.openai_adapter.time.sleep"),
        ):
            with self.assertRaises(Exception):
                self.client.chat.completions.create(model="test-model")
        self.assertTrue(all(k.dead for k in self.manager.keys))


if __name__ == "__main__":
    unittest.main()
