"""
Unit tests for concurrent SQLite database operations.
Tests thread safety of UsageDatabase and AsyncUsageLogger.
"""

import threading
import time

import pytest

from keycycle.usage.db_logic import UsageDatabase
from keycycle.usage.usage_logger import AsyncUsageLogger


class TestSqliteConcurrency:
    """Tests for concurrent SQLite operations."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test_concurrent.db")

    @pytest.fixture
    def db_url(self, db_path):
        """Create database URL from path."""
        return f"sqlite:///{db_path}"

    @pytest.mark.unit
    def test_sqlite_concurrent_logging(self, db_url):
        """Test concurrent read/write operations on SQLite database."""
        db = UsageDatabase(db_url=db_url)
        logger = AsyncUsageLogger(db)

        provider = "openai"
        api_key = "sk-1234567890abcdef"
        errors = []

        def logger_task():
            """Write 10 log entries."""
            try:
                for i in range(10):
                    logger.log(provider, f"model-{i}", api_key, 100 + i)
                    time.sleep(0.1)
            except Exception as e:
                errors.append(f"Logger error: {e}")

        def reader_task():
            """Read history multiple times while logging."""
            try:
                for i in range(5):
                    db.load_history(provider, api_key, seconds_lookback=60)
                    time.sleep(0.2)
            except Exception as e:
                errors.append(f"Reader error: {e}")

        t1 = threading.Thread(target=logger_task)
        t2 = threading.Thread(target=reader_task)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        logger.stop()

        # Check no errors occurred
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"

        # Verify all records were written
        final_history = db.load_history(provider, api_key, seconds_lookback=60)
        assert len(final_history) == 10, f"Expected 10 records, got {len(final_history)}"

    @pytest.mark.unit
    def test_sqlite_multiple_writers(self, db_url):
        """Test multiple concurrent writers."""
        db = UsageDatabase(db_url=db_url)
        logger = AsyncUsageLogger(db)

        provider = "test_provider"
        errors = []

        def writer_task(key_suffix: int):
            """Write entries with a specific key."""
            api_key = f"sk-key-{key_suffix}"
            try:
                for i in range(5):
                    logger.log(provider, f"model-{i}", api_key, 50)
                    time.sleep(0.05)
            except Exception as e:
                errors.append(f"Writer {key_suffix} error: {e}")

        threads = [threading.Thread(target=writer_task, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logger.stop()

        assert len(errors) == 0, f"Multiple writers failed: {errors}"

        # Verify each key has 5 records
        for i in range(3):
            api_key = f"sk-key-{i}"
            history = db.load_history(provider, api_key, seconds_lookback=60)
            assert len(history) == 5, f"Key {i}: expected 5 records, got {len(history)}"

    @pytest.mark.unit
    def test_database_initialization(self, db_url):
        """Test that database initializes correctly."""
        db = UsageDatabase(db_url=db_url)
        assert db is not None

        # Should be able to query empty database
        history = db.load_history("test", "test-key", seconds_lookback=60)
        assert history == []

    @pytest.mark.unit
    def test_logger_stop_flushes(self, db_url):
        """Test that logger.stop() flushes pending writes."""
        db = UsageDatabase(db_url=db_url)
        logger = AsyncUsageLogger(db)

        provider = "flush_test"
        api_key = "sk-flush-test"

        # Log multiple entries quickly
        for i in range(5):
            logger.log(provider, f"model-{i}", api_key, 100)

        # Stop should flush all entries
        logger.stop()

        # All entries should be persisted
        history = db.load_history(provider, api_key, seconds_lookback=60)
        assert len(history) == 5, f"Expected 5 flushed records, got {len(history)}"
