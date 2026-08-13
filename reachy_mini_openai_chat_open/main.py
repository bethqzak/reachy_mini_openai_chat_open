"""Reachy Mini OpenAI Chat app (open edition).

A standalone Reachy Mini app that runs a live voice conversation through the
OpenAI Realtime API using *your own* OpenAI key, while keeping the robot alive:
ambient movement, speech-reactive motion, expressive gestures, camera vision
("what do you see?"), and optional face tracking. Conversations are never
written to disk or uploaded anywhere — the settings page shows a live
transcript from memory only.

Entry point (see pyproject.toml):
    [project.entry-points."reachy_mini_apps"]
    reachy_mini_openai_chat_open = "reachy_mini_openai_chat_open.main:OpenAIChatApp"
"""

from __future__ import annotations

import asyncio
import threading
import time

from reachy_mini import ReachyMini, ReachyMiniApp

from .audio import MicReader, SpeakerPlayer
from .config import Config
from .log import get_logger
from .motion import MotionController
from .realtime import OpenAIRealtimeClient
from .tools import ToolContext
from .transcript import Transcript
from .vision import FaceTracker

logger = get_logger("app")


class OpenAIChatApp(ReachyMiniApp):
    # Settings page served from static/ and embedded in the dashboard.
    # Reachable at http://localhost:8042 (Lite) once the app is running.
    custom_app_url: str | None = "http://0.0.0.0:8042"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        cfg = Config.from_env()
        logger.info(
            "starting: model=%s voice=%s camera=%s face_tracking=%s ambient=%s "
            "half_duplex=%s tools=%s key_set=%s",
            cfg.model, cfg.voice, cfg.enable_camera, cfg.enable_face_tracking,
            cfg.ambient_enabled, cfg.half_duplex, cfg.tools_enabled,
            bool(cfg.openai_api_key),
        )

        media = getattr(reachy_mini, "media", None)
        if media is None:
            logger.error("no media backend on this robot — cannot run")
            raise RuntimeError(
                "This robot has no media backend available (need microphone, "
                "speaker and camera). Start the app with the default media "
                "backend enabled."
            )

        # --- start audio I/O and discover real sample rates ---------------
        try:
            media.start_recording()
            media.start_playing()
        except Exception:
            logger.exception("could not start audio recording/playback")

        cfg.robot_sample_rate = _safe_rate(media, "get_input_audio_samplerate", 16000)
        out_rate = _safe_rate(media, "get_output_audio_samplerate", cfg.robot_sample_rate)
        logger.info("audio rates: mic=%sHz speaker=%sHz openai=%sHz",
                    cfg.robot_sample_rate, out_rate, cfg.openai_sample_rate)

        # --- build the pieces --------------------------------------------
        mic = MicReader(media, cfg.robot_sample_rate, cfg.openai_sample_rate)
        speaker = SpeakerPlayer(media, out_rate, cfg.openai_sample_rate)

        face_tracker = None
        if cfg.enable_camera:
            face_tracker = FaceTracker(media)
            if not face_tracker.available:
                logger.warning("OpenCV face cascade not available; face tracking disabled.")
            else:
                face_tracker.set_enabled(cfg.enable_face_tracking)

        motion = MotionController(reachy_mini, cfg, speaker=speaker,
                                  face_tracker=face_tracker)
        tool_ctx = ToolContext(motion, face_tracker, media, cfg)
        transcript = Transcript(cfg)

        # Shared state for the settings UI + a flag to request a live reconnect.
        self.cfg = cfg
        self.motion = motion
        self.speaker = speaker
        self.face_tracker = face_tracker
        self.transcript = transcript
        self._restart = threading.Event()
        self._runtime = {"connected": False, "greeted": False, "last_error": None}

        # Register the settings-page API routes on self.settings_app.
        try:
            from .webui import register_routes
            register_routes(self)
        except Exception:
            logger.exception("settings UI unavailable")

        speaker.start()
        motion.start()
        if face_tracker is not None:
            face_tracker.start()

        if cfg.openai_api_key:
            logger.info("ready — talk to Reachy Mini! Settings: http://localhost:8042")
        else:
            logger.info("no OpenAI API key yet — open http://localhost:8042 and "
                        "paste your key in the settings page to start chatting.")

        # --- run the realtime session, reconnecting on drops/restarts ----
        try:
            while not stop_event.is_set():
                if not cfg.openai_api_key:
                    # No key yet: idle until one arrives from the settings UI.
                    self._runtime["connected"] = False
                    time.sleep(0.2)
                    continue
                self._restart.clear()
                client = OpenAIRealtimeClient(cfg, mic, speaker, tool_ctx,
                                              logger=transcript,
                                              runtime=self._runtime,
                                              restart_event=self._restart)
                try:
                    asyncio.run(client.run(stop_event))
                except Exception:
                    logger.exception("realtime session crashed")
                if stop_event.is_set():
                    break
                if self._restart.is_set():
                    self._restart.clear()
                    logger.info("applying settings — reconnecting ...")
                    continue  # immediate reconnect with updated config
                logger.info("reconnecting in 2s ...")
                for _ in range(20):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
        finally:
            logger.info("shutting down ...")
            motion.stop()
            speaker.stop()
            if face_tracker is not None:
                face_tracker.stop()
            for _ in range(20):  # let motion settle before releasing
                time.sleep(0.01)
            try:
                media.stop_recording()
                media.stop_playing()
            except Exception:
                logger.exception("error while stopping audio")
            logger.info("shutdown complete.")


def _safe_rate(media, method: str, default: int) -> int:
    try:
        fn = getattr(media, method, None)
        if fn is None:
            logger.debug("media has no %s(); assuming %s Hz", method, default)
            return default
        val = int(fn())
        return val if val > 0 else default
    except Exception as e:
        logger.warning("media.%s() failed (%s); assuming %s Hz", method, e, default)
        return default


if __name__ == "__main__":
    app = OpenAIChatApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
