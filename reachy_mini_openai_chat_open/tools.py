"""Tool (function) definitions exposed to the OpenAI Realtime model, and the
dispatcher that turns a tool call into a robot action.

Keeping the schema and the dispatch together makes it easy to add a new robot
capability: add an entry to TOOL_SCHEMAS and a branch in dispatch().
"""

from __future__ import annotations

from .motion import AVAILABLE_EXPRESSIONS
from .vision import capture_jpeg_data_uri
from .websearch import search

# Tools that can take seconds (network calls). The realtime client runs these
# off its event loop so speech, the mic and barge-in keep working meanwhile.
SLOW_TOOLS = {"web_search"}


def tool_schemas(enable_camera: bool = True,
                 enable_web_search: bool = True) -> list[dict]:
    schemas = [
        {
            "type": "function",
            "name": "express",
            "description": (
                "Move your head, body and antennas to express yourself while "
                "talking. Use this often as natural body language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The movement or emotion to perform.",
                        "enum": AVAILABLE_EXPRESSIONS + ["reset"],
                    }
                },
                "required": ["action"],
            },
        },
        {
            "type": "function",
            "name": "set_face_tracking",
            "description": (
                "Turn face-following on or off. When on, you turn your head to "
                "look at the nearest person's face."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"}
                },
                "required": ["enabled"],
            },
        },
    ]
    if enable_camera:
        schemas.append(
            {
                "type": "function",
                "name": "look",
                "description": (
                    "Capture an image from your camera so you can see what is in "
                    "front of you. Call this whenever the user asks what you see, "
                    "what something is, to read something, or to look at them or "
                    "an object. After looking, describe what you actually see."
                ),
                "parameters": {"type": "object", "properties": {}},
            }
        )
    if enable_web_search:
        schemas.append(
            {
                "type": "function",
                "name": "web_search",
                "description": (
                    "Look something up on the internet. Use it for anything "
                    "current or that may have changed: news, weather, sports "
                    "results, prices, opening hours, recent events, or facts "
                    "you are not sure of. Don't use it for things you already "
                    "know well. Takes a few seconds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A complete, self-contained search question, "
                                "including any place or date the user meant."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            }
        )
    return schemas


class ToolContext:
    """Handles the robot resources a tool needs to act on."""

    def __init__(self, motion, face_tracker, media, config):
        self.motion = motion
        self.face_tracker = face_tracker
        self.media = media
        self.config = config


def dispatch(name: str, args: dict, ctx: ToolContext) -> tuple[str, str | None]:
    """Execute a tool call.

    Returns (output_text, image_data_uri_or_None). The text is sent back as the
    function_call_output; if an image URI is returned the caller should also add
    it to the conversation as an input_image so the model can see it.
    """
    args = args or {}

    if name == "express":
        action = str(args.get("action", "")).strip()
        ok = ctx.motion.trigger(action)
        if not ok:
            return (f'{{"ok": false, "error": "unknown action {action!r}"}}', None)
        return (f'{{"ok": true, "action": "{action}"}}', None)

    if name == "set_face_tracking":
        enabled = bool(args.get("enabled", False))
        if ctx.face_tracker is None or not ctx.face_tracker.available:
            return ('{"ok": false, "error": "face tracking unavailable"}', None)
        ctx.face_tracker.set_enabled(enabled)
        return (f'{{"ok": true, "face_tracking": {str(enabled).lower()}}}', None)

    if name == "look":
        if not ctx.config.enable_camera:
            return ('{"ok": false, "error": "camera disabled"}', None)
        cfg = ctx.config
        uri = capture_jpeg_data_uri(
            ctx.media, blur_faces=getattr(cfg, "blur_faces", True),
            method=cfg.face_method, runtime=cfg.face_runtime,
            style=cfg.blur_style)
        if uri is None:
            return ('{"ok": false, "error": "could not capture image"}', None)
        # Nudge a little "looking" gesture for liveliness.
        ctx.motion.trigger("look_around")
        return ('{"ok": true, "status": "captured"}', uri)

    if name == "web_search":
        if not ctx.config.enable_web_search:
            return ('{"ok": false, "error": "web search disabled"}', None)
        # A "thinking" tilt so the wait reads as the robot looking it up.
        ctx.motion.trigger("tilt_head")
        cfg = ctx.config
        return (search(str(args.get("query", "")), cfg.openai_api_key,
                       cfg.web_search_model), None)

    return (f'{{"ok": false, "error": "unknown tool {name!r}"}}', None)
