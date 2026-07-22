"""Tool (function) definitions exposed to the OpenAI Realtime model, and the
dispatcher that turns a tool call into a robot action.

Keeping the schema and the dispatch together makes it easy to add a new robot
capability: add an entry to TOOL_SCHEMAS and a branch in dispatch().
"""

from __future__ import annotations

from .motion import AVAILABLE_EXPRESSIONS
from .vision import capture_jpeg_data_uri


def tool_schemas(enable_camera: bool = True) -> list[dict]:
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
        uri = capture_jpeg_data_uri(
            ctx.media, blur_faces=getattr(ctx.config, "blur_faces", True))
        if uri is None:
            return ('{"ok": false, "error": "could not capture image"}', None)
        # Nudge a little "looking" gesture for liveliness.
        ctx.motion.trigger("look_around")
        return ('{"ok": true, "status": "captured"}', uri)

    return (f'{{"ok": false, "error": "unknown tool {name!r}"}}', None)
