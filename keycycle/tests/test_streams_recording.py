"""
Integration tests for streaming usage recording.
Tests that both sync and async streams properly record request and token usage.
"""

import asyncio
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / "local.env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ruff: noqa: E402  (the providers read their keys as they import: the env loads first)

from keycycle import MultiProviderWrapper


class TestStreamsRecording:
    """Tests for usage recording during streaming requests."""

    @pytest.fixture
    def cerebras_wrapper(self, load_env):
        """Create Cerebras wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="cerebras", default_model_id="gpt-oss-120b", env_file=str(ENV_PATH)
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
                provider="groq",
                default_model_id="meta-llama/llama-4-maverick-17b-128e-instruct",
                env_file=str(ENV_PATH),
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Groq not configured: {e}")

    @pytest.fixture
    def openrouter_wrapper(self, load_env):
        """Create OpenRouter wrapper."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="openrouter",
                default_model_id="xiaomi/mimo-v2-flash:free",
                env_file=str(ENV_PATH),
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"OpenRouter not configured: {e}")

    async def _run_stream_test(self, wrapper, provider_name: str):
        """Run stream recording test for a provider."""
        initial_stats = wrapper.manager.get_global_stats()
        initial_reqs = initial_stats.total.total_requests
        initial_tokens = initial_stats.total.total_tokens

        # Test 1: Sync Stream
        client = wrapper.get_openai_client()
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Write a one word poem."}],
            max_tokens=10,
            stream=True,
            stream_options={"include_usage": True},
        )
        sync_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                sync_content += chunk.choices[0].delta.content

        assert len(sync_content) > 0, f"{provider_name} sync stream returned no content"

        # Test 2: Async Stream
        aclient = wrapper.get_async_openai_client()
        stream = await aclient.chat.completions.create(
            messages=[{"role": "user", "content": "Write another one word poem."}],
            max_tokens=10,
            stream=True,
        )
        async_content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                async_content += chunk.choices[0].delta.content

        assert len(async_content) > 0, f"{provider_name} async stream returned no content"

        # Small buffer for background processing
        await asyncio.sleep(0.5)

        final_stats = wrapper.manager.get_global_stats()
        final_reqs = final_stats.total.total_requests
        final_tokens = final_stats.total.total_tokens

        req_diff = final_reqs - initial_reqs
        token_diff = final_tokens - initial_tokens

        # We made 2 requests
        assert req_diff >= 2, (
            f"{provider_name}: Expected at least 2 requests recorded, got {req_diff}"
        )

        return req_diff, token_diff

    @pytest.mark.integration
    async def test_cerebras_stream_recording(self, cerebras_wrapper):
        """Test that Cerebras streams record usage properly."""
        req_diff, token_diff = await self._run_stream_test(cerebras_wrapper, "Cerebras")
        # Cerebras should report tokens
        assert token_diff > 0, f"Cerebras: Expected tokens to be recorded, got {token_diff}"

    @pytest.mark.integration
    async def test_groq_stream_recording(self, groq_wrapper):
        """Test that Groq streams record usage properly."""
        req_diff, token_diff = await self._run_stream_test(groq_wrapper, "Groq")
        # Groq should report tokens
        assert token_diff > 0, f"Groq: Expected tokens to be recorded, got {token_diff}"

    @pytest.mark.integration
    async def test_openrouter_stream_recording(self, openrouter_wrapper):
        """Test that OpenRouter streams record usage properly."""
        req_diff, token_diff = await self._run_stream_test(openrouter_wrapper, "OpenRouter")
        # Free models might not report tokens, so we just check requests
        # Token check is a warning for free models
        if token_diff <= 0:
            # This is expected for some free models
            pass

    @pytest.mark.integration
    def test_sync_stream_basic(self, cerebras_wrapper):
        """Test basic sync stream functionality."""
        client = cerebras_wrapper.get_openai_client()

        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = list(stream)
        assert len(chunks) > 0, "No chunks received from sync stream"

    @pytest.mark.integration
    async def test_async_stream_basic(self, cerebras_wrapper):
        """Test basic async stream functionality."""
        aclient = cerebras_wrapper.get_async_openai_client()

        stream = await aclient.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello."}], max_tokens=10, stream=True
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0, "No chunks received from async stream"
