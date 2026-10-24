import os
import sys
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from keycycle.adapters.openai_adapter import BaseRotatingClient
    from keycycle.core.utils import is_rate_limit_error, is_temporary_rate_limit_error
    from keycycle.key_rotation.rotating_mixin import RotatingCredentialsMixin
except ImportError:
    # Fallback for different path structures or if run directly
    from keycycle.keycycle.adapters.openai_adapter import BaseRotatingClient
    from keycycle.keycycle.core.utils import is_rate_limit_error, is_temporary_rate_limit_error
    from keycycle.keycycle.key_rotation.rotating_mixin import RotatingCredentialsMixin


class MockAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        # Some libraries put body in response
        self.response = body


class TestRateLimitLogic(unittest.TestCase):
    def setUp(self):
        # Setup mocks for BaseRotatingClient
        self.mock_manager = MagicMock()
        self.mock_limit_resolver = MagicMock()

        # Instantiate BaseRotatingClient
        # We can instantiate it directly as we are only testing _is_rate_limit_error
        # which doesn't rely on abstract methods in __init__ for this test purpose,
        # but BaseRotatingClient is not abstract in python unless abc is used.
        # It does check HAS_OPENAI, so we assume openai is installed or mocked.
        self.client = BaseRotatingClient(
            manager=self.mock_manager,
            limit_resolver=self.mock_limit_resolver,
            default_model="test-model",
            provider="openai",
        )

        # Setup mocks for RotatingCredentialsMixin
        self.mock_wrapper = MagicMock()
        # RotatingCredentialsMixin expects keyword args including wrapper
        self.mixin = RotatingCredentialsMixin(model_id="test-model", wrapper=self.mock_wrapper)

    def test_openai_adapter_rate_limit_detection_error_1(self):
        """Test detection of the first specific 429 error format (OpenAI/Standard)."""
        # ERROR Rate limit error from OpenAI API: Error code: 429 - {'error': {'message': 'Rate limitexceeded: free-models-per-day...
        error_message = "Error code: 429 - {'error': {'message': 'Rate limitexceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests perday', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '50','X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1768348800000'}, 'provider_name':None}}, 'user_id': 'user_35Uqp8KORjNGZ6rcimjhpReCR1K'}"

        e = Exception(error_message)

        self.assertTrue(is_rate_limit_error(e), f"Failed to detect Error 1. Error str: {str(e)}")

    def test_openai_adapter_rate_limit_detection_error_2(self):
        """Test detection of the second specific 429 error format (Provider returned error / OpenRouter)."""
        # ERROR Rate limit error from OpenAI API: Error code: 429 - {'error': {'message': 'Providerreturned error', ...
        error_message = "Error code: 429 - {'error': {'message': 'Providerreturned error', 'code': 429, 'metadata': {'raw': 'qwen/qwen3-coder:free istemporarily rate-limited upstream. Please retry shortly, or add your own key toaccumulate your rate limits: https://openrouter.ai/settings/integrations','provider_name': 'Venice'}}, 'user_id': 'user_35WM9XjOiswqhxGPhAcjh7dCmq8'}"

        e = Exception(error_message)

        self.assertTrue(is_rate_limit_error(e), f"Failed to detect Error 2. Error str: {str(e)}")

    def test_mixin_rate_limit_detection_error_1(self):
        """Test detection of the first specific 429 error format."""
        error_message = "Error code: 429 - {'error': {'message': 'Rate limitexceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests perday', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '50','X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1768348800000'}, 'provider_name':None}}, 'user_id': 'user_35Uqp8KORjNGZ6rcimjhpReCR1K'}"
        e = Exception(error_message)
        self.assertTrue(is_rate_limit_error(e), "Failed to detect Error 1")

    def test_mixin_rate_limit_detection_error_2(self):
        """
        Test the TRICKY case where str(e) does NOT contain '429',
        but the body/metadata does.
        """
        # 1. The message is generic (this caused the original bug)
        generic_message = "Provider returned error"

        # 2. The details are in the body (reconstructed from your log)
        error_body = {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": "qwen/qwen3-coder:free is temporarily rate-limited upstream...",
                "provider_name": "Venice",
            },
        }

        # 3. Use the Mock class, NOT the base Exception class
        e = MockAPIError(generic_message, body=error_body)

        # Verify assumptions about the test setup itself
        print(f"\nDebug: str(e) is: '{str(e)}'")  # Should print 'Provider returned error'
        assert "429" not in str(e), "Test setup flaw: 429 shouldn't be in the string representation"

        # 4. Run the check
        self.assertTrue(is_rate_limit_error(e), "Failed to detect Error 2 via body inspection")

    def test_status_code_detection(self):
        """Test detection via status_code attribute."""
        e = MockAPIError("Some error", status_code=429)
        self.assertTrue(is_rate_limit_error(e), "Failed status_code check")

    def test_body_detection(self):
        """Test detection via body content."""
        e = MockAPIError("Generic error", body={"message": "You are rate limited"})
        self.assertTrue(is_rate_limit_error(e), "Failed body check")

    def test_keyword_variations(self):
        """Test various keywords that should trigger rate limit detection."""
        keywords = ["too many requests", "resource exhausted", "rate limit", "rate-limited"]

        for kw in keywords:
            e = Exception(f"Some prefix {kw} some suffix")
            self.assertTrue(is_rate_limit_error(e), f"Failed to detect keyword: {kw}")


class TestTemporaryRateLimitDetection(unittest.TestCase):
    """Test cases for distinguishing temporary vs hard rate limits."""

    def test_temporary_rate_limit_openrouter_upstream(self):
        """OpenRouter upstream rate limit should be detected as temporary."""
        error_message = (
            "qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly"
        )
        e = Exception(error_message)

        self.assertTrue(
            is_temporary_rate_limit_error(e),
            "Failed to detect OpenRouter upstream error as temporary",
        )

    def test_hard_rate_limit_daily_quota(self):
        """Daily quota exceeded should NOT be detected as temporary."""
        error_message = "Rate limit exceeded: free-models-per-day"
        e = Exception(error_message)

        self.assertFalse(
            is_temporary_rate_limit_error(e),
            "Daily quota error should NOT be detected as temporary",
        )

    def test_high_traffic_is_temporary(self):
        """High traffic messages should be temporary."""
        error_message = "Server experiencing high traffic. Please retry."
        e = Exception(error_message)

        self.assertTrue(
            is_temporary_rate_limit_error(e), "High traffic error should be detected as temporary"
        )

    def test_retry_shortly_is_temporary(self):
        """'Retry shortly' messages should be temporary."""
        error_message = "Error 429: Please retry shortly"
        e = Exception(error_message)

        self.assertTrue(
            is_temporary_rate_limit_error(e),
            "'Retry shortly' error should be detected as temporary",
        )

    def test_hourly_limit_is_hard(self):
        """Per-hour limits should be hard (not temporary)."""
        error_message = "Rate limit exceeded: requests per-hour"
        e = Exception(error_message)

        self.assertFalse(
            is_temporary_rate_limit_error(e), "Per-hour limit should NOT be detected as temporary"
        )

    def test_quota_exceeded_is_hard(self):
        """Quota exceeded should be hard (not temporary)."""
        error_message = "Error 429: Quota exceeded for this API key"
        e = Exception(error_message)

        self.assertFalse(
            is_temporary_rate_limit_error(e), "Quota exceeded should NOT be detected as temporary"
        )

    def test_non_rate_limit_returns_false(self):
        """Non-rate-limit errors should return False."""
        e = Exception("Connection timeout")

        self.assertFalse(
            is_temporary_rate_limit_error(e), "Non-rate-limit error should return False"
        )

    def test_ambiguous_429_defaults_to_not_temporary(self):
        """A 429 with no clear indicators should default to not temporary (rotate key)."""
        e = Exception("Error code: 429 - Unknown error")

        # Should be a rate limit error, but not temporary
        self.assertFalse(
            is_temporary_rate_limit_error(e),
            "Ambiguous 429 should default to not temporary (rotate key)",
        )

    def test_temporary_with_body_content(self):
        """Temporary indicator in body should be detected."""
        error_body = {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": "Model is temporarily rate-limited upstream",
            },
        }
        e = MockAPIError("Provider returned error", status_code=429, body=error_body)

        self.assertTrue(
            is_temporary_rate_limit_error(e), "Temporary indicator in body should be detected"
        )

    def test_hard_limit_takes_precedence(self):
        """Hard limit indicators should take precedence over temporary ones."""
        # This error has both "retry shortly" (temp) AND "per-day" (hard)
        # Hard should take precedence
        error_message = "Rate limit exceeded: free-models-per-day. Please retry shortly."
        e = Exception(error_message)

        self.assertFalse(
            is_temporary_rate_limit_error(e),
            "Hard limit indicator should take precedence over temporary",
        )


if __name__ == "__main__":
    unittest.main()
