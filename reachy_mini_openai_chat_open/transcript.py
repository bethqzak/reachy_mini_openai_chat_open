"""In-memory transcript for the settings page.

Holds the recent conversation — user speech, the robot's replies, and
tool/vision moments — in a bounded in-memory buffer so the dashboard can show
a live transcript. Nothing is written to disk and nothing leaves the robot;
the buffer disappears when the app stops.
"""

from __future__ import annotations

import base64
import threading
from collections import deque
from datetime import datetime

from .log import get_logger

logger = get_logger("transcript")

MAX_ENTRIES = 200
MAX_IMAGES = 12  # camera thumbnails kept for the transcript view


def _tool_summary(name: str, args: dict) -> str:
    """Human-readable one-liner for a tool/vision event."""
    args = args or {}
    if name == "express":
        return f"Reachy expresses: {args.get('action', '?')}"
    if name == "look":
        return "Reachy looked through its camera"
    if name == "set_face_tracking":
        return f"Reachy face-tracking: {'on' if args.get('enabled') else 'off'}"
    return f"Reachy tool: {name} {args}"


class Transcript:
    """Thread-safe rolling transcript. Same producer API the realtime client
    expects: log_user / log_assistant / log_tool."""

    def __init__(self, cfg=None):
        self._lock = threading.Lock()
        self._entries: deque = deque(maxlen=MAX_ENTRIES)
        self._images: "dict[str, bytes]" = {}
        self._image_order: deque = deque()
        self._image_count = 0

    # -- producer API (called from the realtime client) ----------------- #
    def log_user(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._append("user", text)

    def log_assistant(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._append("assistant", text)

    def log_tool(self, name: str, args: dict, result: str,
                 image_data_uri: str | None = None) -> None:
        meta = {"tool": name, "args": args or {}}
        if image_data_uri:
            try:
                data = base64.b64decode(image_data_uri.split(",", 1)[1])
                with self._lock:
                    self._image_count += 1
                    key = f"img{self._image_count:04d}.jpg"
                    self._images[key] = data
                    self._image_order.append(key)
                    while len(self._image_order) > MAX_IMAGES:
                        self._images.pop(self._image_order.popleft(), None)
                meta["image_file"] = key
            except Exception:
                logger.exception("could not decode camera image for transcript")
        self._append("tool", _tool_summary(name, args), meta=meta)

    # -- consumer API (settings page) ------------------------------------ #
    def entries(self, n: int = 25) -> list[dict]:
        with self._lock:
            items = list(self._entries)
        return items[-max(1, min(n, MAX_ENTRIES)):]

    def image(self, name: str) -> bytes | None:
        with self._lock:
            return self._images.get(name)

    # -- internals ------------------------------------------------------- #
    def _append(self, role: str, text: str, meta: dict | None = None) -> None:
        rec = {
            "ts": datetime.now().astimezone().isoformat(),
            "role": role,
            "text": text,
        }
        if meta:
            rec["meta"] = meta
        with self._lock:
            self._entries.append(rec)
