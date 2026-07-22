"""Central logging for the Reachy Mini OpenAI Chat app.

Every module gets its logger from here so one switch controls the whole app:

    REACHY_LOG_LEVEL=DEBUG    # DEBUG | INFO (default) | WARNING | ERROR
    REACHY_LOG_FILE=~/reachy-chat.log   # optional: also write a DEBUG log file

Console output keeps the familiar `[tag]` prefixes and goes to stdout (which
the Reachy daemon captures). The optional file always logs at DEBUG so a
post-mortem has full detail even when the console is at INFO.

`throttle(key, interval)` gates log lines inside hot loops (audio at ~25 Hz,
motion at 50 Hz) so a persistent failure is reported once per interval
instead of flooding the output.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

_ROOT = "reachy_mini_openai_chat_open"
_configured = False

_throttle_lock = threading.Lock()
_throttle_last: dict[str, float] = {}
_throttle_skipped: dict[str, int] = {}


def get_logger(tag: str) -> logging.Logger:
    """Logger for one module; `tag` shows up as the [tag] console prefix."""
    return logging.getLogger(f"{_ROOT}.{tag}")


def setup_logging() -> None:
    """Configure the app's root logger from the environment. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger(_ROOT)
    level_name = os.environ.get("REACHY_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    fmt = logging.Formatter(
        "%(asctime)s [%(tag)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    console.addFilter(_TagFilter())
    root.addHandler(console)
    root.setLevel(logging.DEBUG)  # handlers decide what gets through
    root.propagate = False

    log_file = os.environ.get("REACHY_LOG_FILE", "").strip()
    if log_file:
        try:
            path = Path(os.path.expanduser(log_file))
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(tag)s] %(levelname)s %(message)s"))
            fh.addFilter(_TagFilter())
            root.addHandler(fh)
            root.info("debug log file: %s", path, extra={"tag": "log"})
        except Exception as e:
            root.warning("could not open REACHY_LOG_FILE %r: %s", log_file, e,
                         extra={"tag": "log"})

    if level_name not in ("INFO", ""):
        root.info("console log level: %s", logging.getLevelName(level),
                  extra={"tag": "log"})


class _TagFilter(logging.Filter):
    """Derive the [tag] prefix from the logger name unless given via extra=."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "tag"):
            record.tag = record.name.rsplit(".", 1)[-1]
        return True


def throttle(key: str, interval: float = 10.0) -> bool:
    """Return True at most once per `interval` seconds for this key.

    Use to rate-limit error logging in hot loops. Suppressed occurrences are
    counted; the count since the last allowed line is available via
    `throttle_skipped(key)` so the log can say "(+N repeats)".
    """
    now = time.monotonic()
    with _throttle_lock:
        last = _throttle_last.get(key)
        if last is not None and (now - last) < interval:
            _throttle_skipped[key] = _throttle_skipped.get(key, 0) + 1
            return False
        _throttle_last[key] = now
        return True


def throttle_skipped(key: str) -> int:
    """Number of suppressed occurrences since the last allowed line; resets."""
    with _throttle_lock:
        return _throttle_skipped.pop(key, 0)
