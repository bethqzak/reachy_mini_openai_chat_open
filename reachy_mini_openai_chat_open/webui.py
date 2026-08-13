"""Settings-page API for the Reachy Mini OpenAI Chat app (open edition).

Routes are registered on `app.settings_app` (the FastAPI instance the SDK
provides). The static page under `static/` calls these to read and change
settings live.

Two tiers of settings:
  * instant   — mutate shared objects / config in place and take effect on the
                next control-loop tick (face tracking, ambient motion,
                half-duplex).
  * reconnect — change the OpenAI session (voice, model, language, personality,
                greeting); these set `app._restart` so the session reconnects
                once with the new config. Conversation context resets, which is
                why they're grouped separately in the UI.

Changes to `config.PERSISTED_FIELDS` (personality, greeting and model) are also
written to the per-user settings file so they are still there after a restart.
"""

# NOTE: no `from __future__ import annotations` here. FastAPI resolves
# stringified annotations against module globals, but `Request` is imported
# inside register_routes — with the future import, every `request: Request`
# parameter degrades into a required query param and all POSTs return 422.
import re

from .config import PERSISTED_FIELDS
from .log import get_logger

log = get_logger("webui")

APP_NAME = "reachy_mini_openai_chat_open"


def _app_version() -> str:
    try:
        from importlib.metadata import version
        return version("reachy-mini-openai-chat-open")
    except Exception:
        return "dev"


def _clamp_volume(value, lo: float = 0.0, hi: float = 2.0) -> float:
    """Coerce a UI-supplied volume multiplier into a safe range.

    Never trust the browser: anything out of range is pinned, and junk falls
    back to unity rather than silencing or deafening the robot.

    2.0 is the ceiling because the model's audio already arrives near
    full-scale — past ~2x the soft limiter absorbs almost all the extra gain,
    so the only thing a higher number buys is distortion.
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 1.0
    if val != val:  # NaN — comparisons are all False, so it would clamp to `hi`
        return 1.0
    return max(lo, min(hi, val))


# Voices supported by the Realtime API (for the UI dropdown).
VOICES = ["marin", "cedar", "alloy", "ash", "ballad", "coral", "echo",
          "sage", "shimmer", "verse"]

# Language is intentionally not settable: always English (config default "en").
RECONNECT_FIELDS = {
    "voice": "voice",
    "model": "model",
    "instructions": "instructions",
    "greeting": "greeting",
}


def register_routes(app) -> None:
    sa = getattr(app, "settings_app", None)
    if sa is None:  # settings server not available in this environment
        log.warning("no settings_app on this SDK; settings page API disabled")
        return

    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse, Response
    except Exception:
        log.warning("fastapi not importable; settings page API disabled")
        return

    def _current_state() -> dict:
        cfg = app.cfg
        ft = app.face_tracker
        face_available = bool(ft is not None and getattr(ft, "available", False))
        speaker = getattr(app, "speaker", None)
        return {
            "app": APP_NAME,
            "version": _app_version(),
            "settings": {
                "voice": cfg.voice,
                "model": cfg.model,
                "instructions": cfg.instructions,
                "greeting": cfg.greeting,
                "face_tracking": bool(ft is not None and ft.enabled),
                "face_available": face_available,
                "ambient": cfg.ambient_enabled,
                "half_duplex": cfg.half_duplex,
                "camera": cfg.enable_camera,
                # Read live off the speaker, not a config field — that's what
                # lets the page tell its own echoed-back value from a stale one.
                "speaker_volume": float(getattr(speaker, "volume", 1.0)),
            },
            "status": {
                "connected": bool(app._runtime.get("connected", False)),
                # Why the last connection attempt failed (e.g. "OpenAI
                # rejected the API key (401)"); empty when none.
                "error": app._runtime.get("last_error") or "",
                # Never send the key itself to the browser — only a hint.
                "key_set": bool(cfg.openai_api_key),
                "key_hint": (cfg.openai_api_key[:7] + "…" + cfg.openai_api_key[-4:]
                             if len(cfg.openai_api_key) > 14 else ""),
            },
            "voices": VOICES,
        }

    @sa.get("/api/state")
    def get_state():
        return _current_state()

    @sa.post("/api/settings")
    async def set_settings(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        cfg = app.cfg
        applied: list[str] = []
        reconnect = False

        # --- instant settings ---
        if "face_tracking" in data:
            val = bool(data["face_tracking"])
            if app.face_tracker is not None and getattr(app.face_tracker, "available", False):
                app.face_tracker.set_enabled(val)
                cfg.enable_face_tracking = val
                applied.append("face_tracking")
        if "ambient" in data:
            cfg.ambient_enabled = bool(data["ambient"])
            applied.append("ambient")
        if "half_duplex" in data:
            cfg.half_duplex = bool(data["half_duplex"])
            applied.append("half_duplex")
        speaker = getattr(app, "speaker", None)
        if "speaker_volume" in data and speaker is not None:
            speaker.volume = _clamp_volume(data["speaker_volume"])
            applied.append("speaker_volume")

        # --- reconnect settings ---
        for key, attr in RECONNECT_FIELDS.items():
            if key in data:
                new = data[key]
                if isinstance(new, str):
                    new = new.strip()
                if new and getattr(cfg, attr) != new:
                    setattr(cfg, attr, new)
                    applied.append(key)
                    reconnect = True

        # Persist the fields that should survive a restart. A failure here
        # only costs the user the change on next launch, so it must not fail
        # the request — the setting is already live for this run.
        saved: list[str] = []
        save_failed = False
        to_save = {k: getattr(cfg, RECONNECT_FIELDS.get(k, k))
                   for k in applied if k in PERSISTED_FIELDS}
        if to_save:
            try:
                from .config import save_settings
                save_settings(to_save)
                saved = sorted(to_save)
            except Exception:
                log.exception("could not persist settings (still active this run)")
                save_failed = True

        if applied:
            log.info("settings changed: %s%s", ", ".join(applied),
                     " (reconnecting)" if reconnect else "")
        if reconnect:
            app._restart.set()

        return {"ok": True, "applied": applied, "reconnect": reconnect,
                "saved": saved, "save_failed": save_failed}

    @sa.post("/api/key")
    async def set_key(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        key = str(data.get("key", "")).strip()
        if not key:
            return JSONResponse({"ok": False, "error": "empty key"}, status_code=400)
        app.cfg.openai_api_key = key
        app._runtime["last_error"] = None  # fresh attempt, fresh verdict
        app._restart.set()  # reconnect (or first connect) with the new key
        log.info("API key updated from settings page; reconnecting")
        persisted = True
        try:
            from .config import save_api_key
            save_api_key(key)
        except Exception:
            log.exception("could not persist API key (still active this run)")
            persisted = False
        return {"ok": True, "persisted": persisted}

    @sa.post("/api/express")
    async def express(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        action = str(data.get("action", "")).strip()
        ok = bool(app.motion.trigger(action)) if action else False
        return {"ok": ok, "action": action}

    @sa.get("/api/image/{name}")
    def image(name: str):
        # Serve a camera capture from the in-memory transcript buffer. Only
        # bare .jpg names generated by this app are accepted — no paths.
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", name):
            return JSONResponse({"error": "not found"}, status_code=404)
        data = app.transcript.image(name)
        if data is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return Response(content=data, media_type="image/jpeg")

    @sa.get("/api/transcript")
    def transcript(n: int = 25):
        return {"entries": app.transcript.entries(n)}
