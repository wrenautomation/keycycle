from unittest.mock import patch

import pytest
from dotenv import load_dotenv

# Import your wrapper class
# Change 'your_module' to the actual path where MultiProviderWrapper is defined
from keycycle import MultiProviderWrapper


# --- Mocks & Fixtures ---
@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Automatically loads local.env before any tests run."""
    load_dotenv(dotenv_path="local.env", override=True)


class MockAgnoModel:
    """A dummy class to simulate an Agno model for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # Mimic attributes usually set by the Mixin or Model
        self.api_key = kwargs.get("api_key")
        self.model_id = kwargs.get("id")


@pytest.fixture
def mock_wrapper():
    """
    Creates a MultiProviderWrapper with 3 fake API keys.
    We patch 'load_api_keys' so we don't need a real .env file, and run with
    track_usage=False so no usage database (TIDB_DB_URL) is needed either.
    """
    fake_keys = ["sk-key-1-alpha", "sk-key-2-beta", "sk-key-3-gamma"]

    with patch.object(MultiProviderWrapper, "load_api_keys", return_value=fake_keys):
        # We also pass a dummy model_class so we don't need 'agno' installed to test logic
        wrapper = MultiProviderWrapper(
            provider="test_provider",
            api_keys=fake_keys,
            default_model_id="test-model",
            model_class=MockAgnoModel,
            track_usage=False,
        )
        return wrapper


# --- Tests ---


def test_get_specific_key_by_index(mock_wrapper):
    """Test retrieving a key using an integer index."""
    # Request Key Index 1 (which is "sk-key-2-beta")
    model = mock_wrapper.get_model(key_id=1)

    assert model.api_key == "sk-key-2-beta"

    # Request Key Index 0
    model = mock_wrapper.get_model(key_id=0)
    assert model.api_key == "sk-key-1-alpha"


def test_get_specific_key_by_string(mock_wrapper):
    """Test retrieving a key using a string suffix or full key."""
    # Request by suffix "gamma" (Key 3)
    model = mock_wrapper.get_model(key_id="gamma")
    assert model.api_key == "sk-key-3-gamma"

    # Request by exact match
    model = mock_wrapper.get_model(key_id="sk-key-1-alpha")
    assert model.api_key == "sk-key-1-alpha"


def test_get_model_with_pinning(mock_wrapper):
    """
    Test that 'pin_key=True' passes the constraint to the instance.
    The instance should have _fixed_key_id set.
    """
    target_id = 1

    # 1. Get model with pinning
    model = mock_wrapper.get_model(key_id=target_id, pin_key=True)

    # Check if correct key was selected
    assert model.api_key == "sk-key-2-beta"

    # Check internal state of the mixin to ensure it knows it's pinned
    # (Accessing protected attribute _fixed_key_id for verification)
    assert hasattr(model, "_fixed_key_id")
    assert model._fixed_key_id == target_id

    # Verify that the Mixin would use this ID if it tried to rotate
    # (Simulating what the Mixin does internally)
    next_key = model._rotate_credentials()
    # Since it is pinned to index 1, next_key should still be Key 2
    assert next_key.api_key == "sk-key-2-beta"


def test_get_model_without_pinning(mock_wrapper):
    """
    Test that 'pin_key=False' (default) allows rotation.
    The instance should have _fixed_key_id as None.
    """
    # 1. Get model with a specific start key, but NOT pinned
    model = mock_wrapper.get_model(key_id=0, pin_key=False)

    # It should start with Key 0
    assert model.api_key == "sk-key-1-alpha"

    # But it should NOT be fixed internally
    assert hasattr(model, "_fixed_key_id")
    assert model._fixed_key_id is None


def test_invalid_key_raises_error(mock_wrapper):
    """Test that requesting a non-existent key raises KeyNotFoundError."""
    # Test index out of bounds
    with pytest.raises(Exception) as excinfo:
        mock_wrapper.get_model(key_id=99)
    assert (
        "KeyNotFoundError" in type(excinfo.value).__name__
        or "not found" in str(excinfo.value).lower()
    )

    # Test non-existent suffix
    with pytest.raises(Exception) as excinfo:
        mock_wrapper.get_model(key_id="non-existent-suffix")
    assert (
        "KeyNotFoundError" in type(excinfo.value).__name__
        or "not found" in str(excinfo.value).lower()
    )
