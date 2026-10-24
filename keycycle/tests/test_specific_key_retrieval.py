from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

from keycycle.legacy_multi_provider_wrapper import MultiProviderWrapper


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Automatically loads local.env before any tests run."""
    load_dotenv(dotenv_path="local.env", override=True)


class TestSpecificKeyRetrieval:
    @pytest.fixture
    def wrapper(self):
        # Mock API keys
        api_keys = ["key-000", "key-111", "key-222", "key-333"]

        # Initialize Wrapper manually to avoid env file dep
        wrapper = MultiProviderWrapper(
            provider="test_provider",
            api_keys=api_keys,
            default_model_id="test-model",
            model_class=MagicMock(),  # Mock model class
            db_url="sqlite:///:memory:",  # Use in-memory DB
        )
        return wrapper

    def test_get_specific_key_by_index(self, wrapper: MultiProviderWrapper):
        """Test retrieving keys by integer index"""
        # Test index 0
        key0 = wrapper.get_api_key(key_id=0)
        assert key0 == "key-000"

        # Test index 2
        key2 = wrapper.get_api_key(key_id=2)
        assert key2 == "key-222"

        # Test last index
        key3 = wrapper.get_api_key(key_id=3)
        assert key3 == "key-333"

    def test_get_specific_key_by_string_suffix(self, wrapper: MultiProviderWrapper):
        """Test retrieving keys by string suffix (last 8 chars logic)"""
        # Test suffix match
        key = wrapper.get_api_key(key_id="key-111")
        assert key == "key-111"

        # Test partial suffix match (if supported by _find_key implementation)
        # The current implementation checks: `if k.api_key == identifier or k.api_key.endswith(identifier)`
        key_partial = wrapper.get_api_key(key_id="222")
        assert key_partial == "key-222"

    def test_invalid_index_raises_error(self, wrapper: MultiProviderWrapper):
        """Test that invalid index raises KeyNotFoundError"""
        with pytest.raises(Exception) as excinfo:
            wrapper.get_api_key(key_id=99)
        assert (
            "KeyNotFoundError" in type(excinfo.value).__name__
            or "not found" in str(excinfo.value).lower()
        )

    def test_invalid_string_raises_error(self, wrapper: MultiProviderWrapper):
        """Test that non-existent key string raises KeyNotFoundError"""
        with pytest.raises(Exception) as excinfo:
            wrapper.get_api_key(key_id="non-existent-key")
        assert (
            "KeyNotFoundError" in type(excinfo.value).__name__
            or "not found" in str(excinfo.value).lower()
        )

    def test_usage_tracking_on_specific_key(self, wrapper: MultiProviderWrapper):
        """Ensure usage is tracked even when retrieving specific keys"""
        # Get key 0
        key, usage_obj = wrapper.get_api_key_with_context(key_id=0, estimated_tokens=100)

        # Check that tokens were reserved
        # usage_obj.buckets[model_id].pending_tokens should be increased
        bucket = usage_obj.buckets["test-model"]
        assert bucket.pending_tokens == 100

        # Commit usage
        wrapper.record_key_usage(key, "test-model", actual_tokens=50, estimated_tokens=100)

        # Check actual usage recorded
        assert bucket.total_tokens == 50
        assert bucket.pending_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__])
