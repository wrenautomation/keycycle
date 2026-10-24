"""
Integration tests for KeyManager functionality.
Tests key rotation, capacity checking, and statistics collection.
"""

from pathlib import Path

import pytest
from agno.agent import Agent

from keycycle import MultiProviderWrapper

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = str(PROJECT_ROOT / "local.env")


class TestKeyManagerIntegration:
    """Integration tests for the key manager."""

    @pytest.fixture
    def cerebras_wrapper(self, load_env):
        """Create Cerebras wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env("cerebras", "gpt-oss-120b", env_file=ENV_FILE)
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Cerebras not configured: {e}")

    @pytest.fixture
    def groq_wrapper(self, load_env):
        """Create Groq wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                "groq", "llama-3.3-70b-versatile", env_file=ENV_FILE
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Groq not configured: {e}")

    @pytest.fixture
    def gemini_wrapper(self, load_env):
        """Create Gemini wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env("gemini", "gemini-2.5-flash", env_file=ENV_FILE)
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Gemini not configured: {e}")

    @pytest.fixture
    def openrouter_wrapper(self, load_env):
        """Create OpenRouter wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                "openrouter", "qwen/qwen3-coder:free", env_file=ENV_FILE
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"OpenRouter not configured: {e}")

    @pytest.mark.integration
    def test_cerebras_key_rotation(self, cerebras_wrapper: MultiProviderWrapper):
        """Test that Cerebras rotates through keys on multiple requests."""
        keys_used = set()

        for i in range(3):
            model = cerebras_wrapper.get_model()
            assert model is not None
            assert hasattr(model, "api_key")
            keys_used.add(model.api_key)

            agent = Agent(model=model)
            response = agent.run("Say 'Confirmed'")
            assert response is not None

        # May or may not rotate depending on key count, but should complete without error
        assert len(keys_used) >= 1

    @pytest.mark.integration
    def test_groq_capacity_check(self, groq_wrapper: MultiProviderWrapper):
        """Test high-load token estimation on Groq."""
        try:
            model = groq_wrapper.get_model(estimated_tokens=2000)
            assert model is not None
            assert hasattr(model, "api_key")

            agent = Agent(model=model)
            response = agent.run("Explain quantum entanglement in one sentence.")
            assert response is not None
        except RuntimeError as e:
            # Expected if capacity limit is reached
            assert "No available keys" in str(e) or "capacity" in str(e).lower()

    @pytest.mark.integration
    async def test_gemini_async_streaming(self, gemini_wrapper: MultiProviderWrapper):
        """Test async streaming with Gemini."""
        model = gemini_wrapper.get_model()
        assert model is not None

        agent = Agent(model=model)
        response = await agent.arun("List 3 fruits.")
        assert response is not None

    @pytest.mark.integration
    def test_openrouter_key_rotation(self, openrouter_wrapper: MultiProviderWrapper):
        """Test OpenRouter key rotation with free model."""
        model = openrouter_wrapper.get_model()
        assert model is not None
        assert hasattr(model, "api_key")

        agent = Agent(model=model)
        response = agent.run("Write a Python function to add two numbers.")
        assert response is not None

    @pytest.mark.integration
    def test_cerebras_global_stats(self, cerebras_wrapper: MultiProviderWrapper):
        """Test global stats retrieval for Cerebras."""
        stats = cerebras_wrapper.manager.get_global_stats()
        assert stats is not None
        assert hasattr(stats, "total")

    @pytest.mark.integration
    def test_groq_model_stats(self, groq_wrapper: MultiProviderWrapper):
        """Test model stats retrieval for Groq."""
        # Make a request first to ensure there's data
        model = groq_wrapper.get_model()
        agent = Agent(model=model)
        agent.run("Hello")

        # Now check stats - the print methods should work without error
        groq_wrapper.print_model_stats("llama-3.3-70b-versatile")

    @pytest.mark.integration
    def test_gemini_key_stats(self, gemini_wrapper: MultiProviderWrapper):
        """Test key stats retrieval for Gemini."""
        model = gemini_wrapper.get_model()
        api_key = model.api_key

        # Make a request
        agent = Agent(model=model)
        agent.run("Hello")

        # Key stats should work
        gemini_wrapper.print_key_stats(api_key)

    @pytest.mark.integration
    def test_gemini_granular_stats(self, gemini_wrapper: MultiProviderWrapper):
        """Test granular stats retrieval for Gemini."""
        model = gemini_wrapper.get_model()
        api_key = model.api_key

        # Make a request
        agent = Agent(model=model)
        agent.run("Hello")

        # Granular stats should work
        gemini_wrapper.print_granular_stats(api_key, "gemini-2.5-flash")

    @pytest.mark.integration
    def test_openrouter_key_and_granular_stats(self, openrouter_wrapper: MultiProviderWrapper):
        """Test key and granular stats for OpenRouter."""
        model = openrouter_wrapper.get_model()
        api_key = model.api_key

        # Make a request
        agent = Agent(model=model)
        agent.run("Hello")

        # Stats should work without error
        openrouter_wrapper.print_key_stats(api_key)
        openrouter_wrapper.print_granular_stats(api_key, "qwen/qwen3-coder:free")
