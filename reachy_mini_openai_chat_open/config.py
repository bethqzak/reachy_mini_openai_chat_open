"""Configuration for the Reachy Mini OpenAI chat app (open edition).

All settings can be provided as environment variables, or placed in a `.env`
file. The `.env` file is looked up (in order) at:

  1. $REACHY_MINI_OPENAI_ENV                          (explicit path)
  2. ./.env                                            (current working directory)
  3. ~/.config/reachy_mini_openai_chat_open/.env       (per-user config)
  4. <this package>/../.env                            (next to the installed app)

The only *required* value is OPENAI_API_KEY — but it does not have to come
from the environment: if it is missing, the app still starts and the user can
paste the key into the settings page, which persists it via `save_api_key()`.

Fields edited on the settings page that should stick (see `PERSISTED_FIELDS`) are saved
to `~/.config/reachy_mini_openai_chat_open/settings.json` and layered on top of
the environment at startup, so an edit made in the UI survives a restart.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .log import get_logger, setup_logging

logger = get_logger("config")


def _load_dotenv() -> Path | None:
    """Populate os.environ from the first .env file we can find.

    Uses python-dotenv when available, otherwise a tiny built-in parser so the
    app still works if the optional dependency is missing. Returns the path
    of the file that was used, or None.
    """
    candidates = []
    explicit = os.environ.get("REACHY_MINI_OPENAI_ENV")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path.home() / ".config" / "reachy_mini_openai_chat_open" / ".env")
    candidates.append(Path(__file__).resolve().parent.parent / ".env")

    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
        return path
    except Exception:
        pass

    # Minimal fallback parser (KEY=VALUE, ignores blanks and #comments).
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    return path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


DEFAULT_INSTRUCTIONS = (
    "You are Reachy Mini, a small, friendly desk robot with a movable head and "
    "two antennas. You are curious, warm, playful and concise. Keep spoken "
    "replies short and conversational, the way a real companion robot would talk. "
    "You can move to express yourself: call the `express` tool to nod, shake your "
    "head, tilt with curiosity, wiggle your antennas, or show an emotion. Use it "
    "naturally and often, as body language while you speak. When someone asks what "
    "you can see, what something is, or to look at something, call the `look` tool "
    "to capture an image from your camera and then describe what you actually see. "
    "You can follow a person's face with the `set_face_tracking` tool. Never "
    "describe these tool calls out loud; just do them as you talk."
)

DEFAULT_GREETING = (
    "Greet the user warmly in one short sentence, introduce yourself as Reachy "
    "Mini, and give a little antenna wiggle."
)

# Config fields the settings page may persist across restarts. Add a name here
# and it is saved and restored automatically — nothing else needs to change.
PERSISTED_FIELDS = ("instructions", "greeting", "model")

SETTINGS_FILE = "settings.json"


def config_dir() -> Path:
    """Per-user config directory (also holds the .env written by save_api_key)."""
    return Path.home() / ".config" / "reachy_mini_openai_chat_open"


def load_saved_settings() -> dict[str, str]:
    """Read persisted settings-page values. Never raises: a corrupt or
    hand-edited file must not stop the robot from starting."""
    path = config_dir() / SETTINGS_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("ignoring unreadable settings file %s", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items()
            if k in PERSISTED_FIELDS and isinstance(v, str) and v.strip()}


def save_settings(values: dict) -> Path:
    """Merge `values` into the persisted settings file and return its path.

    Only `PERSISTED_FIELDS` are stored. Written via a temp file + replace so an
    interrupted save can't leave a half-written file behind.
    """
    merged = load_saved_settings()
    merged.update({k: v for k, v in values.items()
                   if k in PERSISTED_FIELDS and isinstance(v, str) and v.strip()})
    cfg_dir = config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / SETTINGS_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    logger.info("settings saved to %s: %s", path, ", ".join(sorted(values)))
    return path


@dataclass
class Config:
    # --- Required ---
    openai_api_key: str = ""

    # --- Model / voice ---
    model: str = "gpt-realtime"
    voice: str = "marin"
    instructions: str = DEFAULT_INSTRUCTIONS
    greeting: str = DEFAULT_GREETING
    greet_on_start: bool = True
    transcription_language: str = "en"

    # --- Behaviour ---
    enable_camera: bool = True
    blur_faces: bool = True  # pixelate faces before a camera image leaves the robot
    enable_face_tracking: bool = False  # can be toggled at runtime by the model
    half_duplex: bool = False  # mute mic while the robot is talking (anti-echo)

    # --- Audio (robot side is 16 kHz float32; OpenAI Realtime is 24 kHz PCM16) ---
    openai_sample_rate: int = 24000
    robot_sample_rate: int = 16000  # overridden at runtime from the media backend
    # How much speech is kept buffered inside the daemon while the robot
    # talks. The player paces itself so barge-in can cut the voice at once;
    # this is the most that can still play after the cut. Raise it if the
    # voice stutters on a slow host.
    speaker_lead_ms: float = 250.0

    # --- Motion tuning ---
    ambient_enabled: bool = True

    tools_enabled: bool = True

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        env_path = _load_dotenv()
        setup_logging()  # after dotenv so REACHY_LOG_LEVEL/_FILE can live in .env
        if env_path is not None:
            logger.info("loaded settings from %s", env_path)
        else:
            logger.info("no .env file found; using environment/defaults only")
        cfg = cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime"),
            voice=os.environ.get("OPENAI_REALTIME_VOICE", "marin"),
            instructions=os.environ.get("REACHY_INSTRUCTIONS", DEFAULT_INSTRUCTIONS),
            greeting=os.environ.get("REACHY_GREETING", DEFAULT_GREETING),
            greet_on_start=_bool("REACHY_GREET_ON_START", True),
            transcription_language=os.environ.get("REACHY_LANGUAGE", "en"),
            enable_camera=_bool("REACHY_ENABLE_CAMERA", True),
            blur_faces=_bool("REACHY_BLUR_FACES", True),
            enable_face_tracking=_bool("REACHY_FACE_TRACKING", False),
            half_duplex=_bool("REACHY_HALF_DUPLEX", False),
            ambient_enabled=_bool("REACHY_AMBIENT", True),
            speaker_lead_ms=_float("REACHY_SPEAKER_LEAD_MS", 250.0),
        )
        # A value saved from the settings page is a deliberate, later choice
        # than anything in .env, so it wins.
        saved = load_saved_settings()
        for key, value in saved.items():
            setattr(cfg, key, value)
        if saved:
            logger.info("restored saved settings: %s", ", ".join(sorted(saved)))
        return cfg


def save_api_key(key: str) -> Path:
    """Persist an API key entered in the settings UI.

    Written to the per-user .env (already on the lookup path above) so the key
    survives restarts without the user ever editing a file by hand.
    """
    cfg_dir = config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / ".env"
    lines: list[str] = []
    if path.is_file():
        lines = [ln for ln in path.read_text().splitlines()
                 if ln.split("=", 1)[0].strip() != "OPENAI_API_KEY"]
    lines.append(f"OPENAI_API_KEY={key}")
    path.write_text("\n".join(lines) + "\n")
    try:
        path.chmod(0o600)  # key material: owner-only where the OS supports it
    except Exception:
        pass
    os.environ["OPENAI_API_KEY"] = key
    logger.info("API key saved to %s", path)
    return path
