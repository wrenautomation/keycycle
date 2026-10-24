"""
Unit tests for turning usage tracking off (keycycle 0.5.0).

Usage tracking is the database half of the library: it hydrates each key's
counters from `usage_logs` on start and writes a row per call. The in-memory
accounting that actually drives rotation is a separate thing and is always on.

These tests cover the three levels the switch lives at:
  - manager: `RotatingKeyManager(db=None)` / `track_usage=False` / the settable
    `manager.track_usage` attribute,
  - wrapper: `MultiProviderWrapper(..., track_usage=False)` and the
    `track_usage` property that propagates to the managers it owns,
  - call:    `client.chat.completions.create(..., track_usage=False)`.

Everything here runs offline against a sqlite file - no network, no real keys.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from keycycle import MultiClientWrapper, MultiProviderWrapper, RotatingKeyManager
from keycycle.adapters.openai_adapter import RotatingOpenAIClient
from keycycle.config.dataclasses import RateLimits
from keycycle.config.enums import RateLimitStrategy
from keycycle.core.exceptions import ConfigurationError
from keycycle.core.utils import get_key_suffix
from keycycle.usage.db_logic import UsageDatabase

KEYS = ["sk-key-one-AAAAAAAA", "sk-key-two-BBBBBBBB"]
MODEL = "test-model"
PROVIDER = "tracking_test"
LIMITS = RateLimits(100, 6000, 10000, 1000000)


# --- Fixtures / helpers ---------------------------------------------------


@pytest.fixture
def db_url(tmp_path):
    """A throwaway sqlite file (a shared in-memory DB is per-connection)."""
    return f"sqlite:///{tmp_path / 'usage.db'}"


@pytest.fixture
def make_manager():
    """Builds managers and stops them (flushing the writer thread) on teardown."""
    created = []

    def _make(**kwargs):
        manager = RotatingKeyManager(
            api_keys=list(KEYS),
            provider_name=PROVIDER,
            strategy=RateLimitStrategy.PER_MODEL,
            **kwargs,
        )
        created.append(manager)
        return manager

    yield _make
    for manager in created:
        manager.stop()


@pytest.fixture
def stop_wrappers():
    """Registers wrappers so their manager threads are stopped on teardown."""
    created = []
    yield created.append
    for wrapper in created:
        if hasattr(wrapper, "stop"):
            wrapper.stop()
        else:
            wrapper.manager.stop()


def count_usage_rows(db_url: str) -> int:
    """Row count of usage_logs, creating the table if it does not exist yet."""
    UsageDatabase(db_url=db_url)  # ensures the schema exists
    engine = create_engine(db_url)
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM usage_logs")).scalar()


def insert_history_row(db: UsageDatabase, api_key: str, tokens: int = 123) -> None:
    """Write one usage_logs row directly, as a previous process would have."""
    with db.engine.connect() as conn:
        conn.execute(
            db.usage_logs.insert(),
            [
                {
                    "provider": PROVIDER,
                    "model": MODEL,
                    "api_key_suffix": get_key_suffix(api_key),
                    "timestamp": time.time(),
                    "tokens": tokens,
                }
            ],
        )
        conn.commit()


class RecordingOpenAIClient:
    """Fake OpenAI client that records the kwargs each call received."""

    def __init__(self, total_tokens: int = 11):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._total_tokens = total_tokens

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=self._total_tokens))


# --- Manager: no database at all ------------------------------------------


class TestManagerWithoutDatabase:
    """db=None means the manager is purely in-memory: no logger, no writes."""

    def test_no_logger_and_tracking_forced_off(self, make_manager):
        manager = make_manager()

        assert manager.db is None
        assert manager.usage_logger is None
        assert manager.track_usage is False

    def test_track_usage_true_is_forced_off_without_a_db(self, make_manager):
        manager = make_manager(track_usage=True)

        assert manager.track_usage is False
        assert manager.usage_logger is None

    def test_get_key_and_record_usage_update_memory(self, make_manager):
        manager = make_manager()

        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        assert key is not None

        manager.record_usage(key, MODEL, actual_tokens=42, estimated_tokens=10)

        stats = manager.get_global_stats()
        assert stats.total.total_requests == 1
        assert stats.total.total_tokens == 42

    def test_acquire_key_and_record_usage_update_memory(self, make_manager):
        manager = make_manager()

        key = manager.acquire_key(MODEL, LIMITS, estimated_tokens=10)
        assert key is not None

        manager.record_usage(key, MODEL, actual_tokens=7, estimated_tokens=10)

        assert manager.get_global_stats().total.total_tokens == 7

    def test_stop_is_safe_without_a_logger(self, make_manager):
        make_manager().stop()  # must not raise

    def test_explicit_track_usage_true_per_call_raises(self, make_manager):
        manager = make_manager()
        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)

        with pytest.raises(ConfigurationError) as excinfo:
            manager.record_usage(key, MODEL, 5, 10, track_usage=True)

        assert "no usage database is configured" in str(excinfo.value)

    def test_in_memory_accounting_still_happened_before_the_raise(self, make_manager):
        manager = make_manager()
        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)

        with pytest.raises(ConfigurationError):
            manager.record_usage(key, MODEL, 5, 10, track_usage=True)

        assert manager.get_global_stats().total.total_tokens == 5


# --- Manager: database present, switch flipped ----------------------------


class TestManagerWithDatabase:
    def test_tracking_on_writes_a_row(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=True)

        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        manager.record_usage(key, MODEL, actual_tokens=99, estimated_tokens=10)
        manager.stop()

        assert count_usage_rows(db_url) == 1

    def test_tracking_off_writes_nothing(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=False)

        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        manager.record_usage(key, MODEL, actual_tokens=99, estimated_tokens=10)
        manager.stop()

        assert count_usage_rows(db_url) == 0
        # ...but the rotation-driving counters moved all the same.
        assert manager.get_global_stats().total.total_tokens == 99

    def test_per_call_off_beats_attribute_on(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=True)
        assert manager.track_usage is True

        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        manager.record_usage(key, MODEL, 50, 10, track_usage=False)
        manager.stop()

        assert count_usage_rows(db_url) == 0

    def test_per_call_on_beats_attribute_off(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=False)

        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        manager.record_usage(key, MODEL, 50, 10, track_usage=True)
        manager.stop()

        assert count_usage_rows(db_url) == 1

    def test_flipping_the_attribute_off_at_runtime_stops_writes(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=True)

        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        manager.record_usage(key, MODEL, 10, 10)

        manager.track_usage = False
        for _ in range(3):
            key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
            manager.record_usage(key, MODEL, 10, 10)
        manager.stop()

        assert count_usage_rows(db_url) == 1

    def test_flipping_the_attribute_back_on_resumes_writes(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=False)

        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        manager.record_usage(key, MODEL, 10, 10)

        manager.track_usage = True
        key = manager.get_key(MODEL, LIMITS, estimated_tokens=10)
        manager.record_usage(key, MODEL, 10, 10)
        manager.stop()

        assert count_usage_rows(db_url) == 1

    def test_tracking_on_hydrates_history(self, db_url, make_manager):
        db = UsageDatabase(db_url=db_url)
        insert_history_row(db, KEYS[0], tokens=123)

        manager = make_manager(db=db, track_usage=True)

        assert manager.get_global_stats().total.total_tokens == 123

    def test_tracking_off_does_not_consult_the_db(self, db_url, make_manager):
        db = UsageDatabase(db_url=db_url)
        insert_history_row(db, KEYS[0], tokens=123)

        manager = make_manager(db=db, track_usage=False)

        assert manager.get_global_stats().total.total_requests == 0
        assert manager.get_global_stats().total.total_tokens == 0


# --- MultiProviderWrapper --------------------------------------------------


class TestMultiProviderWrapperToggle:
    def test_constructor_needs_no_database(self, monkeypatch, stop_wrappers):
        monkeypatch.delenv("TIDB_DB_URL", raising=False)

        wrapper = MultiProviderWrapper(
            provider=PROVIDER,
            api_keys=list(KEYS),
            default_model_id=MODEL,
            track_usage=False,
        )
        stop_wrappers(wrapper)

        assert wrapper.db is None
        assert wrapper.track_usage is False
        assert wrapper.manager.db is None
        assert wrapper.manager.track_usage is False

    def test_from_env_needs_no_database(self, monkeypatch, stop_wrappers):
        monkeypatch.delenv("TIDB_DB_URL", raising=False)

        with patch.object(MultiProviderWrapper, "load_api_keys", return_value=list(KEYS)):
            wrapper = MultiProviderWrapper.from_env(
                provider=PROVIDER,
                default_model_id=MODEL,
                track_usage=False,
            )
        stop_wrappers(wrapper)

        assert wrapper.db is None
        assert wrapper.track_usage is False
        assert wrapper.manager.usage_logger is None

    def test_missing_url_still_raises_when_tracking_is_on(self, monkeypatch):
        monkeypatch.delenv("TIDB_DB_URL", raising=False)

        with pytest.raises(ValueError):
            MultiProviderWrapper(
                provider=PROVIDER,
                api_keys=list(KEYS),
                default_model_id=MODEL,
            )

    def test_property_propagates_to_the_manager(self, db_url, stop_wrappers):
        wrapper = MultiProviderWrapper(
            provider=PROVIDER,
            api_keys=list(KEYS),
            default_model_id=MODEL,
            db_url=db_url,
        )
        stop_wrappers(wrapper)

        assert wrapper.track_usage is True
        assert wrapper.manager.track_usage is True

        wrapper.track_usage = False
        assert wrapper.manager.track_usage is False

        wrapper.track_usage = True
        assert wrapper.manager.track_usage is True

    def test_enabling_without_a_database_raises(self, monkeypatch, stop_wrappers):
        monkeypatch.delenv("TIDB_DB_URL", raising=False)

        wrapper = MultiProviderWrapper(
            provider=PROVIDER,
            api_keys=list(KEYS),
            default_model_id=MODEL,
            track_usage=False,
        )
        stop_wrappers(wrapper)

        with pytest.raises(ConfigurationError) as excinfo:
            wrapper.track_usage = True

        assert "no usage database is configured" in str(excinfo.value)
        assert wrapper.track_usage is False

    def test_record_key_usage_writes_nothing_when_off(self, monkeypatch, stop_wrappers):
        monkeypatch.delenv("TIDB_DB_URL", raising=False)

        wrapper = MultiProviderWrapper(
            provider=PROVIDER,
            api_keys=list(KEYS),
            default_model_id=MODEL,
            track_usage=False,
        )
        stop_wrappers(wrapper)

        wrapper.record_key_usage(KEYS[0], model_id=MODEL, actual_tokens=25)

        assert wrapper.manager.get_global_stats().total.total_tokens == 25


# --- MultiClientWrapper ----------------------------------------------------


class TestMultiClientWrapperToggle:
    def test_constructor_needs_no_database(self, monkeypatch, stop_wrappers):
        monkeypatch.delenv("TIDB_DB_URL", raising=False)

        wrapper = MultiClientWrapper(track_usage=False)
        stop_wrappers(wrapper)
        wrapper.register_provider(PROVIDER, list(KEYS))

        assert wrapper.db is None
        assert wrapper.track_usage is False
        assert wrapper.get_manager(PROVIDER).track_usage is False
        assert wrapper.get_manager(PROVIDER).usage_logger is None

    def test_property_propagates_to_every_manager(self, db_url, stop_wrappers):
        wrapper = MultiClientWrapper(db_url=db_url)
        stop_wrappers(wrapper)
        wrapper.register_provider("alpha", list(KEYS))
        wrapper.register_provider("beta", list(KEYS))

        wrapper.track_usage = False

        assert all(not m.track_usage for m in wrapper._managers.values())

        wrapper.track_usage = True
        assert all(m.track_usage for m in wrapper._managers.values())

    def test_enabling_without_a_database_raises(self, monkeypatch, stop_wrappers):
        monkeypatch.delenv("TIDB_DB_URL", raising=False)

        wrapper = MultiClientWrapper(track_usage=False)
        stop_wrappers(wrapper)

        with pytest.raises(ConfigurationError):
            wrapper.track_usage = True


# --- OpenAI adapter: per-call override ------------------------------------


class TestOpenAIAdapterPerCallOverride:
    """`track_usage` must never reach the OpenAI SDK, which rejects unknown kwargs."""

    def _client(self, manager):
        return RotatingOpenAIClient(
            manager=manager,
            limit_resolver=lambda model_id, key_suffix=None: LIMITS,
            default_model=MODEL,
            estimated_tokens=10,
            max_retries=1,
            provider="gemini",
        )

    def test_kwarg_is_popped_before_the_real_call(self, make_manager):
        manager = make_manager()
        client = self._client(manager)
        fake = RecordingOpenAIClient()

        with patch.object(client, "_get_fresh_client", return_value=fake):
            result = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                track_usage=False,
            )

        assert result.usage.total_tokens == 11
        assert len(fake.calls) == 1
        assert "track_usage" not in fake.calls[0]
        assert fake.calls[0]["model"] == MODEL

    def test_override_is_threaded_into_record_usage(self, make_manager):
        manager = make_manager()
        client = self._client(manager)
        fake = RecordingOpenAIClient()

        with (
            patch.object(client, "_get_fresh_client", return_value=fake),
            patch.object(manager, "record_usage", wraps=manager.record_usage) as spy,
        ):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                track_usage=False,
            )

        spy.assert_called_once()
        assert spy.call_args.kwargs["track_usage"] is False

    def test_no_override_leaves_the_manager_setting_in_charge(self, make_manager):
        manager = make_manager()
        client = self._client(manager)
        fake = RecordingOpenAIClient()

        with (
            patch.object(client, "_get_fresh_client", return_value=fake),
            patch.object(manager, "record_usage", wraps=manager.record_usage) as spy,
        ):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
            )

        assert spy.call_args.kwargs["track_usage"] is None

    def test_override_off_writes_no_rows(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=True)
        client = self._client(manager)
        fake = RecordingOpenAIClient()

        with patch.object(client, "_get_fresh_client", return_value=fake):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                track_usage=False,
            )
        manager.stop()

        assert count_usage_rows(db_url) == 0
        assert manager.get_global_stats().total.total_tokens == 11

    def test_tracked_call_writes_a_row(self, db_url, make_manager):
        manager = make_manager(db=UsageDatabase(db_url=db_url), track_usage=True)
        client = self._client(manager)
        fake = RecordingOpenAIClient()

        with patch.object(client, "_get_fresh_client", return_value=fake):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
            )
        manager.stop()

        assert count_usage_rows(db_url) == 1
