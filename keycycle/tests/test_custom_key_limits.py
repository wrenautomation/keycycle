import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from keycycle.config.dataclasses import RateLimits
    from keycycle.config.enums import RateLimitStrategy
    from keycycle.key_rotation.rotation_manager import RotatingKeyManager
    from keycycle.legacy_multi_provider_wrapper import MultiProviderWrapper
except ImportError:
    # Fallback for different path structures or if run directly
    from keycycle.keycycle.config.dataclasses import RateLimits
    from keycycle.keycycle.config.enums import RateLimitStrategy
    from keycycle.keycycle.key_rotation.rotation_manager import RotatingKeyManager
    from keycycle.keycycle.legacy_multi_provider_wrapper import MultiProviderWrapper


class TestKeyLimitNormalization(unittest.TestCase):
    """Test the _normalize_key_limits helper method."""

    def setUp(self):
        self.api_keys = [
            "sk-test-key-one-AAAAAAAA",
            "sk-test-key-two-BBBBBBBB",
            "sk-test-key-three-CCCCCCCC",
        ]
        self.mock_db = MagicMock()

    def test_normalize_by_index(self):
        """Test that integer indices are properly normalized to suffixes."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()

        limits = RateLimits(100, 6000, 10000, 1000000)
        key_limits = {0: limits, 1: limits}

        result = wrapper._normalize_key_limits_internal(self.api_keys, key_limits)

        # Suffix for key 0 should be "AAAAAAAA"
        self.assertIn("AAAAAAAA", result)
        # Suffix for key 1 should be "BBBBBBBB"
        self.assertIn("BBBBBBBB", result)
        # Key 2 should NOT be in the result
        self.assertNotIn("CCCCCCCC", result)

    def test_normalize_by_suffix(self):
        """Test that string suffixes are properly matched."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()

        limits = RateLimits(100, 6000, 10000, 1000000)
        # Use partial suffix to match
        key_limits = {"BBBBBBBB": limits}

        result = wrapper._normalize_key_limits_internal(self.api_keys, key_limits)

        self.assertIn("BBBBBBBB", result)
        self.assertEqual(result["BBBBBBBB"], limits)

    def test_normalize_by_full_key(self):
        """Test that full key strings are properly matched."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()

        limits = RateLimits(50, 3000, 5000, 500000)
        key_limits = {"sk-test-key-three-CCCCCCCC": limits}

        result = wrapper._normalize_key_limits_internal(self.api_keys, key_limits)

        self.assertIn("CCCCCCCC", result)
        self.assertEqual(result["CCCCCCCC"], limits)

    def test_normalize_out_of_range_index(self):
        """Test that out-of-range indices are logged as warnings."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()

        limits = RateLimits(100, 6000, 10000)
        key_limits = {99: limits}  # Out of range

        result = wrapper._normalize_key_limits_internal(self.api_keys, key_limits)

        # Result should be empty
        self.assertEqual(result, {})
        # Warning should have been logged
        wrapper.logger.warning.assert_called()

    def test_normalize_unmatched_suffix(self):
        """Test that unmatched suffixes are logged as warnings."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()

        limits = RateLimits(100, 6000, 10000)
        key_limits = {"ZZZZZZZZ": limits}  # Does not match any key

        result = wrapper._normalize_key_limits_internal(self.api_keys, key_limits)

        # Result should be empty
        self.assertEqual(result, {})
        # Warning should have been logged
        wrapper.logger.warning.assert_called()

    def test_normalize_empty_key_limits(self):
        """Test that empty or None key_limits returns empty dict."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()

        self.assertEqual(wrapper._normalize_key_limits_internal(self.api_keys, None), {})
        self.assertEqual(wrapper._normalize_key_limits_internal(self.api_keys, {}), {})


class TestResolveLimits(unittest.TestCase):
    """Test the _resolve_limits method with per-key overrides."""

    def setUp(self):
        self.api_keys = [
            "sk-test-key-one-AAAAAAAA",
            "sk-test-key-two-BBBBBBBB",
        ]

    def test_resolve_with_key_override_single_limit(self):
        """Test resolving limits when a key has a single RateLimits override."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()
        wrapper.default_model_id = "test-model"
        wrapper.provider = "test"

        override_limits = RateLimits(100, 6000, 10000, 1000000)
        wrapper._key_limits = {"AAAAAAAA": override_limits}
        wrapper.MODEL_LIMITS = {"test": {"default": RateLimits(5, 300, 20)}}

        # With key suffix that has override
        result = wrapper._resolve_limits_internal("test-model", "AAAAAAAA")
        self.assertEqual(result, override_limits)

        # With key suffix that has NO override, should fall back to defaults
        result = wrapper._resolve_limits_internal("test-model", "BBBBBBBB")
        self.assertEqual(result.requests_per_minute, 5)

    def test_resolve_with_key_override_per_model(self):
        """Test resolving limits when a key has per-model overrides."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()
        wrapper.default_model_id = "model-a"
        wrapper.provider = "test"

        model_a_limits = RateLimits(100, 6000, 10000)
        model_b_limits = RateLimits(50, 3000, 5000)
        default_for_key = RateLimits(25, 1500, 2500)

        wrapper._key_limits = {
            "AAAAAAAA": {
                "model-a": model_a_limits,
                "model-b": model_b_limits,
                "__default__": default_for_key,
            }
        }
        wrapper.MODEL_LIMITS = {"test": {"default": RateLimits(5, 300, 20)}}

        # Test specific model overrides
        self.assertEqual(wrapper._resolve_limits_internal("model-a", "AAAAAAAA"), model_a_limits)
        self.assertEqual(wrapper._resolve_limits_internal("model-b", "AAAAAAAA"), model_b_limits)

        # Test __default__ fallback for unknown model on this key
        self.assertEqual(wrapper._resolve_limits_internal("model-c", "AAAAAAAA"), default_for_key)

    def test_resolve_without_key_suffix(self):
        """Test resolving limits when no key_suffix is provided."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()
        wrapper.default_model_id = "test-model"
        wrapper.provider = "test"
        wrapper._key_limits = {"AAAAAAAA": RateLimits(100, 6000, 10000)}
        wrapper.MODEL_LIMITS = {"test": {"default": RateLimits(5, 300, 20)}}

        # Without key suffix, should use provider defaults
        result = wrapper._resolve_limits_internal("test-model", None)
        self.assertEqual(result.requests_per_minute, 5)

    def test_resolve_no_overrides(self):
        """Test resolving limits when no key overrides exist."""
        wrapper = MultiProviderWrapper.__new__(MultiProviderWrapper)
        wrapper.logger = MagicMock()
        wrapper.default_model_id = "test-model"
        wrapper.provider = "test"
        wrapper._key_limits = {}
        wrapper.MODEL_LIMITS = {"test": {"test-model": RateLimits(10, 600, 100)}}

        result = wrapper._resolve_limits_internal("test-model", "AAAAAAAA")
        self.assertEqual(result.requests_per_minute, 10)


class TestRotatingKeyManagerWithLimitResolver(unittest.TestCase):
    """Test RotatingKeyManager using the limit_resolver callback."""

    def setUp(self):
        self.api_keys = [
            "sk-test-key-one-AAAAAAAA",
            "sk-test-key-two-BBBBBBBB",
        ]
        self.mock_db = MagicMock()
        self.mock_db.load_provider_history.return_value = []

    def test_get_key_uses_limit_resolver(self):
        """Test that get_key calls the limit_resolver for per-key limits."""
        # Track which key's suffix was passed to the resolver
        resolver_calls = []

        def mock_resolver(model_id, key_suffix):
            resolver_calls.append((model_id, key_suffix))
            # Return very high limits so all keys pass
            return RateLimits(1000, 10000, 100000)

        manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=self.mock_db,
            limit_resolver=mock_resolver,
        )

        default_limits = RateLimits(5, 300, 20)
        manager.get_key("test-model", default_limits, estimated_tokens=100)

        # Should have called the resolver
        self.assertTrue(len(resolver_calls) > 0)
        # First call should be for the first key's suffix
        self.assertEqual(resolver_calls[0][0], "test-model")
        self.assertEqual(resolver_calls[0][1], "AAAAAAAA")

    def test_get_key_without_limit_resolver(self):
        """Test that get_key uses default_limits when no resolver is provided."""
        manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=self.mock_db,
            limit_resolver=None,
        )

        # Very restrictive limits that will block after first use
        restrictive_limits = RateLimits(1, 1, 1)

        # First key should be available
        key = manager.get_key("test-model", restrictive_limits, estimated_tokens=100)
        self.assertIsNotNone(key)

    def test_per_key_limits_affect_key_selection(self):
        """Test that per-key limits actually affect which key is selected."""

        def mock_resolver(model_id, key_suffix):
            if key_suffix == "AAAAAAAA":
                # First key has exhausted limits (0 rpm)
                return RateLimits(0, 0, 0)
            else:
                # Second key has available limits
                return RateLimits(100, 6000, 10000)

        manager = RotatingKeyManager(
            api_keys=self.api_keys,
            provider_name="test",
            strategy=RateLimitStrategy.PER_MODEL,
            db=self.mock_db,
            limit_resolver=mock_resolver,
        )

        default_limits = RateLimits(5, 300, 20)
        key = manager.get_key("test-model", default_limits, estimated_tokens=100)

        # Should skip the first key (exhausted) and return the second
        self.assertIsNotNone(key)
        self.assertTrue(key.api_key.endswith("BBBBBBBB"))


class TestFromEnvWithTiers(unittest.TestCase):
    """Test from_env factory method with tier configuration."""

    @patch.object(MultiProviderWrapper, "load_api_keys")
    @patch.object(MultiProviderWrapper, "__init__", return_value=None)
    def test_tier_config_converts_to_key_limits(self, mock_init, mock_load_keys):
        """Test that key_tiers and tier_limits are converted to key_limits."""
        mock_load_keys.return_value = ["key1", "key2", "key3"]

        free_limits = RateLimits(5, 300, 20, 250000)
        pro_limits = RateLimits(100, 6000, 1000, 1000000)

        MultiProviderWrapper.from_env(
            provider="gemini",
            default_model_id="gemini-2.5-flash",
            key_tiers={0: "free", 1: "pro", 2: "free"},
            tier_limits={
                "free": free_limits,
                "pro": pro_limits,
            },
        )

        # Check that __init__ was called with converted key_limits
        call_kwargs = mock_init.call_args[1]
        key_limits = call_kwargs.get("key_limits")

        self.assertIsNotNone(key_limits)
        self.assertEqual(key_limits[0], free_limits)
        self.assertEqual(key_limits[1], pro_limits)
        self.assertEqual(key_limits[2], free_limits)

    @patch.object(MultiProviderWrapper, "load_api_keys")
    @patch.object(MultiProviderWrapper, "__init__", return_value=None)
    def test_key_limits_and_tiers_combined(self, mock_init, mock_load_keys):
        """Test that explicit key_limits are merged with tier config."""
        mock_load_keys.return_value = ["key1", "key2"]

        explicit_limits = RateLimits(200, 12000, 2000)
        tier_limits_obj = RateLimits(100, 6000, 1000)

        MultiProviderWrapper.from_env(
            provider="gemini",
            default_model_id="gemini-2.5-flash",
            key_limits={0: explicit_limits},  # Explicit override for key 0
            key_tiers={1: "pro"},
            tier_limits={"pro": tier_limits_obj},
        )

        call_kwargs = mock_init.call_args[1]
        key_limits = call_kwargs.get("key_limits")

        # Key 0 should have explicit limits
        self.assertEqual(key_limits[0], explicit_limits)
        # Key 1 should have tier limits
        self.assertEqual(key_limits[1], tier_limits_obj)


if __name__ == "__main__":
    unittest.main()
