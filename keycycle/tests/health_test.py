"""
Integration tests for provider API key health checking.
Validates that configured API keys can make successful requests.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / "local.env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ruff: noqa: E402  (the providers read their keys as they import: the env loads first)

from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.google.gemini import Gemini
from agno.models.groq import Groq
from agno.models.openrouter import OpenRouter


class TestProviderKeyHealth:
    """Tests to verify API keys are valid and working for each provider."""

    def _check_provider_key(self, prefix: str, provider_class, model_id: str, key_index: int):
        """
        Helper to check a single provider key.
        Returns tuple of (success: bool, key_env_name: str, error: str | None)
        """
        key_env = f"{prefix}_API_KEY_{key_index}"
        key = os.getenv(key_env)

        if not key:
            return False, key_env, "Key not found in environment"

        try:
            agent = Agent(model=provider_class(id=model_id, api_key=key), markdown=True)
            response = agent.run("Say 'OK'.")
            response_text = str(response)

            if "429" in response_text or "quota" in response_text.lower():
                return False, key_env, "Rate limit or quota exceeded"

            return True, key_env, None
        except Exception as e:
            return False, key_env, str(e)

    @pytest.mark.integration
    @pytest.mark.parametrize("key_index", range(1, 6))  # Test first 5 keys
    def test_groq_key_health(self, key_index, load_env):
        """Test Groq API key health."""
        num_keys = int(os.getenv("NUM_GROQ", 0))
        if key_index > num_keys:
            pytest.skip(f"GROQ key {key_index} not configured (NUM_GROQ={num_keys})")

        success, key_env, error = self._check_provider_key(
            "GROQ", Groq, "llama-3.3-70b-versatile", key_index
        )
        assert success, f"{key_env} failed: {error}"

    @pytest.mark.integration
    @pytest.mark.parametrize("key_index", range(1, 6))  # Test first 5 keys
    def test_cerebras_key_health(self, key_index, load_env):
        """Test Cerebras API key health."""
        num_keys = int(os.getenv("NUM_CEREBRAS", 0))
        if key_index > num_keys:
            pytest.skip(f"CEREBRAS key {key_index} not configured (NUM_CEREBRAS={num_keys})")

        success, key_env, error = self._check_provider_key(
            "CEREBRAS", Cerebras, "gpt-oss-120b", key_index
        )
        assert success, f"{key_env} failed: {error}"

    @pytest.mark.integration
    @pytest.mark.parametrize("key_index", range(1, 6))  # Test first 5 keys
    def test_gemini_key_health(self, key_index, load_env):
        """Test Gemini API key health."""
        num_keys = int(os.getenv("NUM_GEMINI", 0))
        if key_index > num_keys:
            pytest.skip(f"GEMINI key {key_index} not configured (NUM_GEMINI={num_keys})")

        success, key_env, error = self._check_provider_key(
            "GEMINI", Gemini, "gemini-2.5-flash", key_index
        )
        assert success, f"{key_env} failed: {error}"

    @pytest.mark.integration
    @pytest.mark.parametrize("key_index", range(1, 6))  # Test first 5 keys
    def test_openrouter_key_health(self, key_index, load_env):
        """Test OpenRouter API key health."""
        num_keys = int(os.getenv("NUM_OPENROUTER", 0))
        if key_index > num_keys:
            pytest.skip(f"OPENROUTER key {key_index} not configured (NUM_OPENROUTER={num_keys})")

        success, key_env, error = self._check_provider_key(
            "OPENROUTER", OpenRouter, "xiaomi/mimo-v2-flash:free", key_index
        )
        assert success, f"{key_env} failed: {error}"

    @pytest.mark.integration
    def test_at_least_one_provider_configured(self, load_env):
        """Verify at least one provider has keys configured."""
        providers = ["GROQ", "CEREBRAS", "GEMINI", "OPENROUTER"]
        configured = [p for p in providers if int(os.getenv(f"NUM_{p}", 0)) > 0]

        assert len(configured) > 0, (
            "No providers configured. Set NUM_<PROVIDER> and <PROVIDER>_API_KEY_N in local.env"
        )
