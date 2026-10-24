"""
Integration tests for environment-based MultiProviderWrapper initialization.
Tests wrapper creation, model retrieval, and stats reporting for multiple providers.
"""

from pathlib import Path

import pytest

from keycycle import MultiProviderWrapper

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = str(PROJECT_ROOT / "local.env")


class TestEnvironmentIntegration:
    """Tests for MultiProviderWrapper initialization from environment."""

    @pytest.fixture
    def cerebras_wrapper(self, load_env):
        """Create Cerebras wrapper from environment."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="cerebras",
                default_model_id="gpt-oss-120b",
                env_file=ENV_PATH,
                temperature=0.7,
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Cerebras not configured: {e}")

    @pytest.fixture
    def groq_wrapper(self, load_env):
        """Create Groq wrapper from environment."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="groq",
                default_model_id="llama-3.3-70b-versatile",
                env_file=ENV_PATH,
                top_p=0.95,
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Groq not configured: {e}")

    @pytest.fixture
    def gemini_wrapper(self, load_env):
        """Create Gemini wrapper from environment."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="gemini", default_model_id="gemini-2.5-flash", env_file=ENV_PATH, top_k=10
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Gemini not configured: {e}")

    @pytest.fixture
    def openrouter_wrapper(self, load_env):
        """Create OpenRouter wrapper from environment."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="openrouter",
                default_model_id="nvidia/nemotron-nano-12b-v2-vl:free",
                env_file=ENV_PATH,
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"OpenRouter not configured: {e}")

    @pytest.mark.integration
    def test_cerebras_wrapper_initialization(self, cerebras_wrapper):
        """Test Cerebras wrapper initializes correctly."""
        assert cerebras_wrapper is not None
        assert cerebras_wrapper.provider == "cerebras"
        assert cerebras_wrapper.default_model_id == "gpt-oss-120b"

    @pytest.mark.integration
    def test_groq_wrapper_initialization(self, groq_wrapper):
        """Test Groq wrapper initializes correctly."""
        assert groq_wrapper is not None
        assert groq_wrapper.provider == "groq"
        assert groq_wrapper.default_model_id == "llama-3.3-70b-versatile"

    @pytest.mark.integration
    def test_gemini_wrapper_initialization(self, gemini_wrapper):
        """Test Gemini wrapper initializes correctly."""
        assert gemini_wrapper is not None
        assert gemini_wrapper.provider == "gemini"
        assert gemini_wrapper.default_model_id == "gemini-2.5-flash"

    @pytest.mark.integration
    def test_openrouter_wrapper_initialization(self, openrouter_wrapper):
        """Test OpenRouter wrapper initializes correctly."""
        assert openrouter_wrapper is not None
        assert openrouter_wrapper.provider == "openrouter"

    @pytest.mark.integration
    def test_cerebras_model_retrieval(self, cerebras_wrapper):
        """Test getting a model from Cerebras wrapper."""
        model = cerebras_wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
        assert model is not None
        assert hasattr(model, "api_key")
        assert model.api_key is not None

    @pytest.mark.integration
    def test_groq_model_retrieval(self, groq_wrapper):
        """Test getting a model from Groq wrapper."""
        model = groq_wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
        assert model is not None
        assert hasattr(model, "api_key")
        assert model.api_key is not None

    @pytest.mark.integration
    def test_gemini_model_retrieval(self, gemini_wrapper):
        """Test getting a model from Gemini wrapper."""
        model = gemini_wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
        assert model is not None
        assert hasattr(model, "api_key")
        assert model.api_key is not None

    @pytest.mark.integration
    def test_openrouter_model_retrieval(self, openrouter_wrapper):
        """Test getting a model from OpenRouter wrapper."""
        model = openrouter_wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
        assert model is not None
        assert hasattr(model, "api_key")
        assert model.api_key is not None

    @pytest.mark.integration
    def test_cerebras_stats_available(self, cerebras_wrapper):
        """Test that Cerebras stats methods are available."""
        # Just verify the method runs without error
        stats = cerebras_wrapper.manager.get_global_stats()
        assert stats is not None

    @pytest.mark.integration
    def test_groq_stats_available(self, groq_wrapper):
        """Test that Groq stats methods are available."""
        stats = groq_wrapper.manager.get_global_stats()
        assert stats is not None

    @pytest.mark.integration
    def test_gemini_stats_available(self, gemini_wrapper):
        """Test that Gemini stats methods are available."""
        stats = gemini_wrapper.manager.get_global_stats()
        assert stats is not None

    @pytest.mark.integration
    def test_openrouter_stats_available(self, openrouter_wrapper):
        """Test that OpenRouter stats methods are available."""
        stats = openrouter_wrapper.manager.get_global_stats()
        assert stats is not None
