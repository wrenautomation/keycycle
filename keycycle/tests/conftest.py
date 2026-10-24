"""
Shared pytest fixtures and configuration for keycycle tests.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Path to local.env for integration tests
LOCAL_ENV_PATH = Path(__file__).parent / "local.env"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests requiring real API keys")
    config.addinivalue_line("markers", "unit: marks unit tests without external dependencies")
    config.addinivalue_line("markers", "slow: marks slow-running tests")
    if LOCAL_ENV_PATH.exists():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=LOCAL_ENV_PATH, override=True)


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip integration tests when API keys are not available.
    """
    skip_integration = pytest.mark.skip(
        reason="Integration tests require API keys (local.env not found or NUM_* not set)"
    )

    for item in items:
        if "integration" in item.keywords:
            # Check if we have any API keys configured
            if not LOCAL_ENV_PATH.exists():
                item.add_marker(skip_integration)
            elif not any(
                os.getenv(f"NUM_{p}")
                for p in ["COHERE", "GROQ", "GEMINI", "CEREBRAS", "OPENROUTER"]
            ):
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def load_env():
    """Load environment variables from local.env if it exists."""
    if LOCAL_ENV_PATH.exists():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=LOCAL_ENV_PATH, override=True)
        return True
    return False


@pytest.fixture
def mock_api_keys() -> list[str]:
    """Provide mock API keys for unit testing."""
    return [
        "sk-test-key-1-alpha-xxxxxxxxxxxxxxxx",
        "sk-test-key-2-beta-xxxxxxxxxxxxxxxx",
        "sk-test-key-3-gamma-xxxxxxxxxxxxxxxx",
    ]


@pytest.fixture
def mock_db():
    """Create a mock UsageDatabase for unit tests."""
    mock = MagicMock()
    mock.load_provider_history.return_value = []
    mock.load_history.return_value = []
    return mock


@pytest.fixture
def mock_rate_limits():
    """Create standard rate limits for testing."""
    from keycycle.config.dataclasses import RateLimits

    return RateLimits(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=1000,
        tokens_per_minute=10000,
        tokens_per_hour=100000,
        tokens_per_day=1000000,
    )


@pytest.fixture
def mock_key_usage():
    """Create a mock KeyUsage object for testing."""
    from keycycle.config.dataclasses import KeyUsage
    from keycycle.config.enums import RateLimitStrategy

    return KeyUsage(
        api_key="sk-test-key-mock-xxxxxxxxxxxxxxxx", strategy=RateLimitStrategy.PER_MODEL
    )


@pytest.fixture(scope="session")
def integration_env():
    """
    Session-scoped fixture that loads local.env once.
    Returns a dict indicating which providers have keys configured.
    """
    from dotenv import load_dotenv

    if LOCAL_ENV_PATH.exists():
        load_dotenv(dotenv_path=LOCAL_ENV_PATH, override=True)

    providers = {}
    for provider in ["COHERE", "GROQ", "GEMINI", "CEREBRAS", "OPENROUTER"]:
        num_keys = os.getenv(f"NUM_{provider}")
        providers[provider.lower()] = int(num_keys) if num_keys else 0

    return providers


def has_provider_keys(provider: str) -> bool:
    """Helper to check if a provider has API keys configured."""
    num = os.getenv(f"NUM_{provider.upper()}")
    return bool(num and int(num) > 0)


# Skip decorators for use in test files
skip_without_cohere = pytest.mark.skipif(
    not has_provider_keys("cohere"), reason="Requires COHERE API keys in local.env"
)

skip_without_groq = pytest.mark.skipif(
    not has_provider_keys("groq"), reason="Requires GROQ API keys in local.env"
)

skip_without_gemini = pytest.mark.skipif(
    not has_provider_keys("gemini"), reason="Requires GEMINI API keys in local.env"
)
