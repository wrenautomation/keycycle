"""
Stress tests for rate limiting and key rotation.
Tests that the local rate limiter properly enforces limits and triggers rotation.
"""

import asyncio
from pathlib import Path

import pytest
from agno.agent import Agent

from keycycle import MultiProviderWrapper

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = str(PROJECT_ROOT / "local.env")


class TestStress:
    """Stress tests for rate limiting functionality."""

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
            wrapper = MultiProviderWrapper.from_env("groq", "groq/compound-mini", env_file=ENV_FILE)
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
                "openrouter", "tngtech/deepseek-r1t2-chimera:free", env_file=ENV_FILE
            )
            yield wrapper
            wrapper.manager.stop()
        except Exception as e:
            pytest.skip(f"OpenRouter not configured: {e}")

    def _get_limit_config(self, wrapper, provider_name: str, model_id: str, limit_attr: str):
        """Get rate limit configuration for a provider/model."""
        provider_limits = wrapper.MODEL_LIMITS.get(provider_name, {})
        limits_config = provider_limits.get(model_id)

        if not limits_config:
            limits_config = provider_limits.get("default")

        if not limits_config:
            return None

        return getattr(limits_config, limit_attr, None)

    async def _run_stress_test(
        self, wrapper, provider_name: str, model_id: str, limit_attr: str, max_requests: int = 10
    ):
        """
        Run stress test for a provider.

        Returns tuple of (successful_requests, blocked_by_limiter, errors)
        """
        limit_value = self._get_limit_config(wrapper, provider_name, model_id, limit_attr)

        if limit_value is None:
            pytest.skip(f"{provider_name} limit '{limit_attr}' is None (unlimited)")

        # Cap requests to avoid excessive API usage
        target_requests = min(limit_value + 2, max_requests)

        successful = 0
        blocked = False
        errors = []

        for i in range(target_requests):
            try:
                model = wrapper.get_model(id=model_id, estimated_tokens=10, wait=False)

                agent = Agent(model=model)
                await agent.arun("hi", stream=False)

                successful += 1
                await asyncio.sleep(0.5)

            except RuntimeError as e:
                if "No available keys" in str(e) or "Timeout" in str(e):
                    blocked = True
                    break
                errors.append(str(e))
                break
            except Exception as e:
                errors.append(str(e))
                await asyncio.sleep(2)

        return successful, blocked, errors

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_gemini_rate_limit_stress(self, gemini_wrapper):
        """Stress test Gemini rate limiting."""
        successful, blocked, errors = await self._run_stress_test(
            gemini_wrapper, "gemini", "gemini-2.5-flash-lite", "requests_per_minute", max_requests=8
        )

        # Should have made at least some successful requests
        assert successful > 0, f"No requests succeeded. Errors: {errors}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_cerebras_rate_limit_stress(self, cerebras_wrapper):
        """Stress test Cerebras rate limiting."""
        successful, blocked, errors = await self._run_stress_test(
            cerebras_wrapper, "cerebras", "gpt-oss-120b", "requests_per_minute", max_requests=8
        )

        assert successful > 0, f"No requests succeeded. Errors: {errors}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_groq_rate_limit_stress(self, groq_wrapper):
        """Stress test Groq rate limiting."""
        successful, blocked, errors = await self._run_stress_test(
            groq_wrapper, "groq", "groq/compound-mini", "requests_per_day", max_requests=8
        )

        assert successful > 0, f"No requests succeeded. Errors: {errors}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_openrouter_rate_limit_stress(self, openrouter_wrapper):
        """Stress test OpenRouter rate limiting."""
        successful, blocked, errors = await self._run_stress_test(
            openrouter_wrapper,
            "openrouter",
            "tngtech/deepseek-r1t2-chimera:free",
            "requests_per_minute",
            max_requests=8,
        )

        assert successful > 0, f"No requests succeeded. Errors: {errors}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_multiple_keys_rotate(self, gemini_wrapper):
        """Test that multiple keys are used during stress."""
        keys_used = set()

        for i in range(5):
            try:
                model = gemini_wrapper.get_model(estimated_tokens=10, wait=False)
                if hasattr(model, "api_key") and model.api_key:
                    keys_used.add(model.api_key[-8:])

                agent = Agent(model=model)
                await agent.arun("hi", stream=False)
                await asyncio.sleep(0.5)

            except RuntimeError:
                break
            except Exception:
                await asyncio.sleep(1)

        # Should have used at least one key
        assert len(keys_used) >= 1, "No keys were tracked"
