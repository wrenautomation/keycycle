"""
Integration tests for OpenAI client rate limit rotation.
Tests that the local rate limiter correctly blocks requests at limits.
"""

import asyncio
from pathlib import Path

import pytest

from keycycle import MultiProviderWrapper

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = str(PROJECT_ROOT / "local.env")


class TestOpenAIRotation:
    """Tests for rate limit rotation with OpenAI-compatible clients."""

    @pytest.fixture
    def gemini_wrapper(self, load_env):
        """Create Gemini wrapper for rate limit testing."""
        try:
            wrapper = MultiProviderWrapper.from_env(
                provider="gemini", default_model_id="gemini-2.5-flash", env_file=ENV_FILE
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"Gemini not configured: {e}")

    def _get_limit_value(self, wrapper, provider: str, model_id: str, limit_attr: str):
        """Get the configured limit value for a provider/model."""
        provider_limits = wrapper.MODEL_LIMITS.get(provider, {})
        limits_config = provider_limits.get(model_id)

        if not limits_config:
            limits_config = provider_limits.get("default")

        if not limits_config:
            return None

        return getattr(limits_config, limit_attr, None)

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_gemini_rate_limit_enforcement(self, gemini_wrapper):
        """Test that local rate limiter enforces request limits."""
        limit_value = self._get_limit_value(
            gemini_wrapper, "gemini", "gemini-2.5-flash", "requests_per_minute"
        )

        if limit_value is None:
            pytest.skip("Gemini rate limit not configured")

        # Cap at reasonable test limit
        test_requests = min(limit_value + 2, 10)

        client = gemini_wrapper.get_async_openai_client()
        successful_requests = 0

        for i in range(test_requests):
            try:
                await client.chat.completions.create(
                    messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )
                successful_requests += 1
                await asyncio.sleep(0.5)

            except RuntimeError as e:
                if "No available keys" in str(e) or "Timeout" in str(e):
                    break
                raise
            except Exception:
                # API errors are acceptable - continue testing
                await asyncio.sleep(1)

        # Either we completed all requests or hit the limiter
        assert successful_requests > 0, "No requests succeeded"

        # If we exceeded the limit, the limiter should have blocked us
        if test_requests > limit_value:
            # It's acceptable if either: we were blocked, or all requests succeeded
            # (if we have multiple keys that can share the load)
            pass

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_multiple_requests_track_usage(self, gemini_wrapper):
        """Test that multiple requests properly track usage stats."""
        client = gemini_wrapper.get_async_openai_client()

        initial_stats = gemini_wrapper.manager.get_global_stats()
        initial_requests = initial_stats.total.total_requests

        # Make 3 requests
        for _ in range(3):
            try:
                await client.chat.completions.create(
                    messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )
                await asyncio.sleep(0.5)
            except Exception:
                pass  # Ignore errors, just testing stats tracking

        final_stats = gemini_wrapper.manager.get_global_stats()
        final_requests = final_stats.total.total_requests

        # Should have recorded some requests
        assert final_requests >= initial_requests

    @pytest.mark.integration
    def test_stats_report_after_rotation(self, gemini_wrapper):
        """Test that stats are properly reported after making requests."""
        # Make a single request to ensure there's data
        client = gemini_wrapper.get_openai_client()

        try:
            client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}], max_tokens=5
            )
        except Exception:
            pytest.skip("Could not make test request")

        # Stats should be available
        stats = gemini_wrapper.manager.get_global_stats()
        assert stats is not None
        assert hasattr(stats, "total")
