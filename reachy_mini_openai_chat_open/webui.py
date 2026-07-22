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
"""

# NOTE: no `from __future__ import annotations` here. FastAPI resolves
# stringified annotations against module globals, but `Request` is imported
# inside register_routes — with the future import, every `request: Request`
# parameter degrades into a required query param and all POSTs return 422.
import re

from .log import get_logger

log = get_logger("webui")

APP_NAME = "reachy_mini_openai_chat_open"


def _app_version() -> str:
    try:
        from importlib.metadata import version
        return version("reachy-mini-openai-chat-open")
    except Exception:
        return "dev"


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

        if applied:
            log.info("settings changed: %s%s", ", ".join(applied),
                     " (reconnecting)" if reconnect else "")
        if reconnect:
            app._restart.set()

        return {"ok": True, "applied": applied, "reconnect": reconnect}

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
