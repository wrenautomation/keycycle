"""
Tests for the new MultiClientWrapper class.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from keycycle import (
    MultiClientWrapper,
    ProviderEnvConfig,
)
from keycycle.config.enums import RateLimitStrategy
from keycycle.core.utils import normalize_key_entries, normalize_key_entry


class TestKeyEntryNormalization(unittest.TestCase):
    """Test the KeyEntry normalization functions."""

    def test_normalize_string_entry(self):
        """Test normalizing a simple string API key."""
        primary, params = normalize_key_entry("sk-test-key-123")

        self.assertEqual(primary, "sk-test-key-123")
        self.assertEqual(params, {"api_key": "sk-test-key-123"})

    def test_normalize_dict_entry(self):
        """Test normalizing a dict entry with extra params."""
        entry = {"api_key": "sk-test-key-123", "index_id": "idx_abc", "org_id": "org_xyz"}
        primary, params = normalize_key_entry(entry)

        self.assertEqual(primary, "sk-test-key-123")
        self.assertEqual(params["api_key"], "sk-test-key-123")
        self.assertEqual(params["index_id"], "idx_abc")
        self.assertEqual(params["org_id"], "org_xyz")

    def test_normalize_dict_entry_custom_api_key_param(self):
        """Test normalizing a dict entry with custom api_key parameter name."""
        entry = {"access_token": "token-123", "project_id": "proj_abc"}
        primary, params = normalize_key_entry(entry, api_key_param="access_token")

        self.assertEqual(primary, "token-123")
        self.assertEqual(params["access_token"], "token-123")
        self.assertEqual(params["project_id"], "proj_abc")

    def test_normalize_dict_missing_api_key_raises(self):
        """Test that dict without api_key raises ValueError."""
        entry = {"index_id": "idx_abc"}

        with self.assertRaises(ValueError) as ctx:
            normalize_key_entry(entry)

        self.assertIn("api_key", str(ctx.exception))

    def test_normalize_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with self.assertRaises(TypeError):
            normalize_key_entry(123)

    def test_normalize_list_of_entries(self):
        """Test normalizing a list of mixed entries."""
        entries = [
            "sk-key-1",
            {"api_key": "sk-key-2", "index_id": "idx_a"},
            "sk-key-3",
        ]

        result = normalize_key_entries(entries)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("sk-key-1", {"api_key": "sk-key-1"}))
        self.assertEqual(result[1][0], "sk-key-2")
        self.assertEqual(result[1][1]["index_id"], "idx_a")
        self.assertEqual(result[2], ("sk-key-3", {"api_key": "sk-key-3"}))


class TestKeyUsageParams(unittest.TestCase):
    """Test the KeyUsage.params field and get_client_params method."""

    def test_key_usage_with_params(self):
        """Test KeyUsage with params field."""
        from keycycle.config.dataclasses import KeyUsage

        params = {"api_key": "sk-test", "index_id": "idx_abc"}
        key_usage = KeyUsage(api_key="sk-test", strategy=RateLimitStrategy.PER_MODEL, params=params)

        client_params = key_usage.get_client_params()

        self.assertEqual(client_params["api_key"], "sk-test")
        self.assertEqual(client_params["index_id"], "idx_abc")

    def test_key_usage_without_params(self):
        """Test KeyUsage without params falls back to api_key only."""
        from keycycle.config.dataclasses import KeyUsage

        key_usage = KeyUsage(
            api_key="sk-test",
            strategy=RateLimitStrategy.PER_MODEL,
        )

        client_params = key_usage.get_client_params()

        self.assertEqual(client_params, {"api_key": "sk-test"})


class TestMultiClientWrapperRegister(unittest.TestCase):
    """Test MultiClientWrapper registration functionality."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.load_provider_history.return_value = []

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_register_single_provider(self, mock_db_class):
        """Test registering a single provider."""
        mock_db_class.return_value = self.mock_db

        wrapper = MultiClientWrapper()
        wrapper.register_provider(
            provider="anthropic",
            keys=["sk-key-1", "sk-key-2"],
            default_model="claude-3-sonnet",
        )

        self.assertIn("anthropic", wrapper._managers)
        self.assertEqual(wrapper._configs["anthropic"].default_model, "claude-3-sonnet")

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_register_provider_with_extra_params(self, mock_db_class):
        """Test registering a provider with extra params (dict keys)."""
        mock_db_class.return_value = self.mock_db

        wrapper = MultiClientWrapper()
        wrapper.register_provider(
            provider="twelvelabs",
            keys=[
                {"api_key": "tl-key-1", "index_id": "idx_abc"},
                {"api_key": "tl-key-2", "index_id": "idx_xyz"},
            ],
            default_model="pegasus-1",
        )

        manager = wrapper._managers["twelvelabs"]

        # Check that params are stored in KeyUsage
        self.assertEqual(manager.keys[0].params["api_key"], "tl-key-1")
        self.assertEqual(manager.keys[0].params["index_id"], "idx_abc")
        self.assertEqual(manager.keys[1].params["index_id"], "idx_xyz")

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_register_multiple_providers(self, mock_db_class):
        """Test registering multiple providers."""
        mock_db_class.return_value = self.mock_db

        wrapper = MultiClientWrapper()
        wrapper.register_provider("anthropic", ["sk-1"], default_model="claude-3")
        wrapper.register_provider("openai", ["sk-2"], default_model="gpt-4")
        wrapper.register_provider("gemini", ["sk-3"], default_model="gemini-pro")

        self.assertEqual(len(wrapper._managers), 3)
        self.assertIn("anthropic", wrapper._managers)
        self.assertIn("openai", wrapper._managers)
        self.assertIn("gemini", wrapper._managers)

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_register_provider_chaining(self, mock_db_class):
        """Test that register_provider returns self for chaining."""
        mock_db_class.return_value = self.mock_db

        wrapper = MultiClientWrapper()
        result = wrapper.register_provider("test", ["key1"]).register_provider("test2", ["key2"])

        self.assertIs(result, wrapper)
        self.assertEqual(len(wrapper._managers), 2)


class TestMultiClientWrapperFromEnv(unittest.TestCase):
    """Test MultiClientWrapper.from_env factory method."""

    @patch.dict(
        os.environ,
        {
            "NUM_ANTHROPIC": "2",
            "ANTHROPIC_API_KEY_1": "sk-ant-1",
            "ANTHROPIC_API_KEY_2": "sk-ant-2",
            "NUM_TWELVELABS": "2",
            "TWELVELABS_API_KEY_1": "tl-key-1",
            "TWELVELABS_INDEX_ID_1": "idx_abc",
            "TWELVELABS_API_KEY_2": "tl-key-2",
            "TWELVELABS_INDEX_ID_2": "idx_xyz",
        },
    )
    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    @patch("keycycle.core.utils.load_dotenv")
    def test_from_env_multiple_providers(self, mock_dotenv, mock_db_class):
        """Test creating wrapper from environment for multiple providers."""
        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.load_provider_history.return_value = []

        wrapper = MultiClientWrapper.from_env(
            {
                "anthropic": ProviderEnvConfig(default_model="claude-3-sonnet"),
                "twelvelabs": ProviderEnvConfig(
                    default_model="pegasus-1",
                    extra_params=["index_id"],
                ),
            }
        )

        self.assertIn("anthropic", wrapper._managers)
        self.assertIn("twelvelabs", wrapper._managers)

        # Check anthropic has simple string keys
        ant_manager = wrapper._managers["anthropic"]
        self.assertEqual(ant_manager.keys[0].api_key, "sk-ant-1")

        # Check twelvelabs has dict keys with extra params
        tl_manager = wrapper._managers["twelvelabs"]
        self.assertEqual(tl_manager.keys[0].params["api_key"], "tl-key-1")
        self.assertEqual(tl_manager.keys[0].params["index_id"], "idx_abc")

    @patch.dict(
        os.environ,
        {
            "NUM_CUSTOM": "1",
            "CUSTOM_API_KEY_1": "custom-key-1",
            "CUSTOM_ORG_ID_1": "org_abc",
            "CUSTOM_PROJECT_ID_1": "proj_xyz",
        },
    )
    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    @patch("keycycle.core.utils.load_dotenv")
    def test_from_env_with_multiple_extra_params(self, mock_dotenv, mock_db_class):
        """Test loading multiple extra params from environment."""
        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.load_provider_history.return_value = []

        wrapper = MultiClientWrapper.from_env(
            {
                "custom": ProviderEnvConfig(
                    default_model="custom-model",
                    extra_params=["org_id", "project_id"],
                ),
            }
        )

        manager = wrapper._managers["custom"]
        params = manager.keys[0].params

        self.assertEqual(params["api_key"], "custom-key-1")
        self.assertEqual(params["org_id"], "org_abc")
        self.assertEqual(params["project_id"], "proj_xyz")


class TestMultiClientWrapperGetClient(unittest.TestCase):
    """Test MultiClientWrapper.get_rotating_client method."""

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_get_client_unregistered_provider_raises(self, mock_db_class):
        """Test that getting client for unregistered provider raises ValueError."""
        mock_db_class.return_value = MagicMock()

        wrapper = MultiClientWrapper()

        with self.assertRaises(ValueError) as ctx:
            wrapper.get_rotating_client("unknown", MagicMock)

        self.assertIn("not registered", str(ctx.exception))

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    @patch("keycycle.multi_client_wrapper.create_rotating_client")
    def test_get_client_uses_default_model(self, mock_create, mock_db_class):
        """Test that get_rotating_client uses provider's default model."""
        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.load_provider_history.return_value = []
        mock_create.return_value = MagicMock()

        wrapper = MultiClientWrapper()
        wrapper.register_provider("test", ["key1"], default_model="test-model-v1")

        mock_client_class = MagicMock
        wrapper.get_rotating_client("test", mock_client_class)

        # Check that create_rotating_client was called with the default model
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["default_model"], "test-model-v1")


class TestLoadApiKeysStatic(unittest.TestCase):
    """Test the static load_api_keys method."""

    @patch.dict(
        os.environ,
        {
            "NUM_TEST": "3",
            "TEST_API_KEY_1": "key-1",
            "TEST_API_KEY_2": "key-2",
            "TEST_API_KEY_3": "key-3",
        },
    )
    @patch("keycycle.core.utils.load_dotenv")
    def test_load_simple_keys(self, mock_dotenv):
        """Test loading simple string keys."""
        keys = MultiClientWrapper.load_api_keys("test")

        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], "key-1")
        self.assertEqual(keys[1], "key-2")
        self.assertEqual(keys[2], "key-3")

    @patch.dict(
        os.environ,
        {
            "NUM_PROVIDER": "2",
            "PROVIDER_API_KEY_1": "key-a",
            "PROVIDER_INDEX_ID_1": "idx-a",
            "PROVIDER_API_KEY_2": "key-b",
            "PROVIDER_INDEX_ID_2": "idx-b",
        },
    )
    @patch("keycycle.core.utils.load_dotenv")
    def test_load_keys_with_extra_params(self, mock_dotenv):
        """Test loading keys with extra params."""
        keys = MultiClientWrapper.load_api_keys("provider", extra_params=["index_id"])

        self.assertEqual(len(keys), 2)
        self.assertIsInstance(keys[0], dict)
        self.assertEqual(keys[0]["api_key"], "key-a")
        self.assertEqual(keys[0]["index_id"], "idx-a")
        self.assertEqual(keys[1]["api_key"], "key-b")
        self.assertEqual(keys[1]["index_id"], "idx-b")

    @patch.dict(os.environ, {"NUM_MISSING": "1"}, clear=False)
    @patch("keycycle.core.utils.load_dotenv")
    def test_load_keys_missing_key_raises(self, mock_dotenv):
        """Test that missing key raises ValueError."""
        # NUM_MISSING is set but MISSING_API_KEY_1 is not
        with self.assertRaises(ValueError) as ctx:
            MultiClientWrapper.load_api_keys("missing")

        self.assertIn("Missing API key", str(ctx.exception))

    @patch.dict(
        os.environ,
        {
            # The fleet convention (0.3.2): the first NUM indices are the
            # sanctioned live keys; keys parked at higher indices beyond a gap
            # are dead or spare ON PURPOSE (mirrors William's GROQ fleet:
            # 1..24 live, 37..46 parked dead). NUM is authoritative.
            "NUM_GAPPY": "24",
            **{f"GAPPY_API_KEY_{i}": f"key-{i}" for i in range(1, 25)},
            **{f"GAPPY_API_KEY_{i}": f"key-{i}" for i in range(37, 47)},
        },
    )
    @patch("keycycle.core.utils.load_dotenv")
    def test_load_keys_num_caps_out_parked_keys(self, mock_dotenv):
        """Only the first NUM sorted indices load; parked keys stay out."""
        keys = MultiClientWrapper.load_api_keys("gappy")

        self.assertEqual(len(keys), 24)
        self.assertEqual(keys[0], "key-1")
        self.assertEqual(keys[-1], "key-24")

    @patch.dict(
        os.environ,
        {
            # A gap INSIDE the sanctioned range is tolerated: index 2 retired,
            # NUM still says 3, so the loader takes the first 3 that exist
            # (1, 3, 4) rather than crashing or stopping at the gap.
            "NUM_HOLEY": "3",
            "HOLEY_API_KEY_1": "key-1",
            "HOLEY_API_KEY_3": "key-3",
            "HOLEY_API_KEY_4": "key-4",
            "HOLEY_API_KEY_9": "key-9",
        },
    )
    @patch("keycycle.core.utils.load_dotenv")
    def test_load_keys_tolerates_gap_inside_sanctioned_range(self, mock_dotenv):
        keys = MultiClientWrapper.load_api_keys("holey")
        self.assertEqual(keys, ["key-1", "key-3", "key-4"])

    @patch.dict(
        os.environ,
        {
            "NUM_SPARSE": "5",
            "SPARSE_API_KEY_1": "key-1",
            "SPARSE_API_KEY_3": "key-3",
        },
    )
    @patch("keycycle.core.utils.load_dotenv")
    def test_load_keys_logs_warning_on_count_mismatch(self, mock_dotenv):
        """A mismatch between NUM_{PROVIDER} and keys found is logged loudly."""
        with self.assertLogs("keycycle.core.utils", level="WARNING") as ctx:
            keys = MultiClientWrapper.load_api_keys("sparse")

        self.assertEqual(keys, ["key-1", "key-3"])
        self.assertTrue(any("NUM_SPARSE" in msg for msg in ctx.output))


class TestKwargFiltering(unittest.TestCase):
    """Test the kwarg filtering functionality (introspection + manual exclusion)."""

    def test_get_valid_constructor_kwargs_explicit_params(self):
        """Test introspection of client with explicit parameters."""
        from keycycle.adapters.generic_adapter import get_valid_constructor_kwargs

        class StrictClient:
            def __init__(self, api_key: str, timeout: int = 30):
                pass

        valid = get_valid_constructor_kwargs(StrictClient)

        self.assertIsNotNone(valid)
        self.assertEqual(valid, frozenset({"api_key", "timeout"}))

    def test_get_valid_constructor_kwargs_with_var_kwargs(self):
        """Test introspection returns None when client accepts **kwargs."""
        from keycycle.adapters.generic_adapter import get_valid_constructor_kwargs

        class FlexibleClient:
            def __init__(self, api_key: str, **kwargs):
                pass

        valid = get_valid_constructor_kwargs(FlexibleClient)

        self.assertIsNone(valid)  # Can't filter because accepts **kwargs

    def test_get_valid_constructor_kwargs_keyword_only(self):
        """Test introspection includes keyword-only parameters."""
        from keycycle.adapters.generic_adapter import get_valid_constructor_kwargs

        class KeywordOnlyClient:
            def __init__(self, api_key: str, *, timeout: int = 30, retry: bool = True):
                pass

        valid = get_valid_constructor_kwargs(KeywordOnlyClient)

        self.assertIsNotNone(valid)
        self.assertEqual(valid, frozenset({"api_key", "timeout", "retry"}))

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_excluded_kwargs_filters_params(self, mock_db_class):
        """Test that excluded_kwargs filters out specified params."""
        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.load_provider_history.return_value = []

        # Track what kwargs were passed to the client
        received_kwargs = {}

        class MockClient:
            def __init__(self, **kwargs):
                received_kwargs.update(kwargs)

        wrapper = MultiClientWrapper()
        wrapper.register_provider(
            provider="test",
            keys=[{"api_key": "test-key", "index_id": "idx_123"}],
            default_model="test-model",
        )

        client = wrapper.get_rotating_client(
            "test",
            MockClient,
            excluded_kwargs=["model", "timeout"],
            model="should-be-excluded",
            timeout=30,
            other_param="should-remain",
        )

        # Trigger client creation by accessing an attribute
        manager = wrapper._managers["test"]
        key_usage = manager.keys[0]

        # Directly call _get_fresh_client to test filtering
        client._get_fresh_client(key_usage)

        self.assertNotIn("model", received_kwargs)
        self.assertNotIn("timeout", received_kwargs)
        self.assertIn("other_param", received_kwargs)
        self.assertEqual(received_kwargs["other_param"], "should-remain")

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_introspection_filters_unknown_kwargs(self, mock_db_class):
        """Test that introspection filters kwargs not in constructor."""
        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.load_provider_history.return_value = []

        received_kwargs = {}

        class StrictClient:
            def __init__(self, api_key: str, timeout: int = 30):
                received_kwargs.update({"api_key": api_key, "timeout": timeout})

        wrapper = MultiClientWrapper()
        wrapper.register_provider(
            provider="test",
            keys=["test-key"],
            default_model="test-model",
        )

        client = wrapper.get_rotating_client(
            "test",
            StrictClient,
            timeout=60,
            unknown_param="should-be-filtered",
            organization="also-filtered",
        )

        # Trigger client creation
        manager = wrapper._managers["test"]
        key_usage = manager.keys[0]
        client._get_fresh_client(key_usage)

        self.assertIn("api_key", received_kwargs)
        self.assertIn("timeout", received_kwargs)
        self.assertEqual(received_kwargs["timeout"], 60)
        # unknown_param and organization should not be in received_kwargs
        # (filtered by introspection)
        self.assertNotIn("unknown_param", received_kwargs)
        self.assertNotIn("organization", received_kwargs)

    @patch("keycycle.multi_client_wrapper.UsageDatabase")
    def test_excluded_kwargs_applied_before_introspection(self, mock_db_class):
        """Test that manual exclusion is applied before introspection."""
        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.load_provider_history.return_value = []

        # Track exactly what kwargs were passed to __init__
        passed_kwargs = []

        class StrictClient:
            def __init__(self, api_key: str, timeout: int = 30):
                passed_kwargs.append(set(locals().keys()) - {"self"})

        wrapper = MultiClientWrapper()
        wrapper.register_provider(
            provider="test",
            keys=["test-key"],
            default_model="test-model",
        )

        # timeout is valid per introspection, but we explicitly exclude it
        client = wrapper.get_rotating_client(
            "test",
            StrictClient,
            excluded_kwargs=["timeout"],
            timeout=60,  # This should be excluded
        )

        manager = wrapper._managers["test"]
        manager.keys[0]

        # Directly check what final_kwargs would be by inspecting the config
        # The excluded_kwargs should filter out timeout
        self.assertIn("timeout", client.config.excluded_kwargs)

        # Also verify the config has valid_kwargs from introspection
        self.assertIsNotNone(client.config.valid_kwargs)
        self.assertEqual(client.config.valid_kwargs, frozenset({"api_key", "timeout"}))

        # And verify excluded_kwargs contains what we specified
        self.assertEqual(client.config.excluded_kwargs, frozenset({"timeout"}))


class TestProviderEnvConfigExcludedKwargs(unittest.TestCase):
    """Test that ProviderEnvConfig properly stores excluded_kwargs."""

    def test_provider_env_config_excluded_kwargs_default(self):
        """Test that excluded_kwargs defaults to None."""
        config = ProviderEnvConfig(default_model="test-model")
        self.assertIsNone(config.excluded_kwargs)

    def test_provider_env_config_excluded_kwargs_set(self):
        """Test that excluded_kwargs can be set."""
        config = ProviderEnvConfig(
            default_model="test-model",
            excluded_kwargs=["model", "timeout"],
        )
        self.assertEqual(config.excluded_kwargs, ["model", "timeout"])


if __name__ == "__main__":
    unittest.main()
