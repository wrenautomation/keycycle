"""
Integration tests for OpenAI-compatible client interface.
Tests sync/async clients and streaming across multiple providers.
"""

from pathlib import Path

import pytest

from keycycle import MultiProviderWrapper

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = str(PROJECT_ROOT / "local.env")


class TestOpenAIClientInterface:
    """Tests for OpenAI-compatible client interface."""

    @pytest.fixture
    def openrouter_wrapper(self, load_env):
        """Create OpenRouter wrapper with free model."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="openrouter",
                default_model_id="mistralai/devstral-2512:free",
                env_file=ENV_PATH,
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"OpenRouter not configured: {e}")

    @pytest.fixture
    def cerebras_wrapper(self, load_env):
        """Create Cerebras wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="cerebras", default_model_id="gpt-oss-120b", env_file=ENV_PATH
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Cerebras not configured: {e}")

    @pytest.fixture
    def groq_wrapper(self, load_env):
        """Create Groq wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="groq", default_model_id="llama-3.1-8b-instant", env_file=ENV_PATH
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Groq not configured: {e}")

    @pytest.mark.integration
    def test_openrouter_sync_client(self, openrouter_wrapper):
        """Test sync OpenAI client with OpenRouter."""
        client = openrouter_wrapper.get_openai_client()
        assert client is not None

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, what model are you? (Sync)"}],
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @pytest.mark.integration
    async def test_openrouter_async_client(self, openrouter_wrapper):
        """Test async OpenAI client with OpenRouter."""
        async_client = openrouter_wrapper.get_async_openai_client()
        assert async_client is not None

        response = await async_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, what model are you? (Async)"}],
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @pytest.mark.integration
    async def test_openrouter_async_stream(self, openrouter_wrapper):
        """Test async streaming with OpenRouter."""
        async_client = openrouter_wrapper.get_async_openai_client()

        stream = await async_client.chat.completions.create(
            messages=[{"role": "user", "content": "Count to 5"}], stream=True
        )

        chunks_received = 0
        content = ""
        async for chunk in stream:
            chunks_received += 1
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        assert chunks_received > 0, "No chunks received from stream"
        assert len(content) > 0, "No content received from stream"

    @pytest.mark.integration
    def test_cerebras_sync_client(self, cerebras_wrapper):
        """Test sync OpenAI client with Cerebras."""
        client = cerebras_wrapper.get_openai_client()

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @pytest.mark.integration
    async def test_cerebras_async_client(self, cerebras_wrapper):
        """Test async OpenAI client with Cerebras."""
        async_client = cerebras_wrapper.get_async_openai_client()

        response = await async_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @pytest.mark.integration
    def test_groq_sync_client(self, groq_wrapper):
        """Test sync OpenAI client with Groq."""
        client = groq_wrapper.get_openai_client()

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @pytest.mark.integration
    async def test_groq_async_client(self, groq_wrapper):
        """Test async OpenAI client with Groq."""
        async_client = groq_wrapper.get_async_openai_client()

        response = await async_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @pytest.mark.integration
    def test_stats_available_after_request(self, openrouter_wrapper):
        """Test that usage stats are available after making requests."""
        client = openrouter_wrapper.get_openai_client()

        # Make a request
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Stats should be accessible
        stats = openrouter_wrapper.manager.get_global_stats()
        assert stats is not None
