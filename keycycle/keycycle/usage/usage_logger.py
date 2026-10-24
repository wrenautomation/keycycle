import atexit
import logging
import queue
import threading
import time
from typing import Optional

from ..config.constants import USAGE_LOG_BATCH_SIZE
from ..config.log_config import default_logger
from ..core.utils import get_key_suffix
from .db_logic import UsageDatabase


# --- ASYNC LOGGER ---
class AsyncUsageLogger:
    """Decouples Turso DB writes from the main thread using batching."""

    def __init__(self, db: UsageDatabase, logger: Optional[logging.Logger] = None):
        self.db = db
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        atexit.register(self.stop)

        self.logger = logger or default_logger

    def log(self, provider: str, model: str, api_key: str, tokens: int):
        self.queue.put((provider, model, api_key, time.time(), tokens))

    def _writer_loop(self):
        batch = []
        insert_stmt = self.db.usage_logs.insert()

        while not self._stop_event.is_set() or not self.queue.empty():  # always empty queue
            try:
                record = self.queue.get(timeout=1.0)
                # Parse record for batch formatting
                provider, model, full_key, ts, tokens = record
                suffix = get_key_suffix(full_key)
                batch.append(
                    {
                        "provider": provider,
                        "model": model,
                        "api_key_suffix": suffix,
                        "timestamp": ts,
                        "tokens": tokens,
                    }
                )

                # Drain queue up to USAGE_LOG_BATCH_SIZE items to batch write
                while len(batch) < USAGE_LOG_BATCH_SIZE:
                    try:
                        r = self.queue.get_nowait()
                        p, m, k, t, tok = r
                        s = get_key_suffix(k)
                        batch.append(
                            {
                                "provider": p,
                                "model": m,
                                "api_key_suffix": s,
                                "timestamp": t,
                                "tokens": tok,
                            }
                        )
                    except queue.Empty:
                        break

                if batch:
                    with self.db.engine.connect() as conn:
                        conn.execute(insert_stmt, batch)
                        conn.commit()
                    batch.clear()

            except queue.Empty:
                continue
            except Exception as e:
                # print(f"Logging thread error: {e}")
                self.logger.exception("Logging thread error", exc_info=e)
                time.sleep(2)

        if batch:
            try:
                with self.db.engine.connect() as conn:
                    conn.execute(insert_stmt, batch)
                    conn.commit()
            except Exception as e:
                self.logger.exception("Logging thread error on exit", exc_info=e)

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                self.logger.warning("AsyncUsageLogger thread did not exit cleanly within timeout.")
