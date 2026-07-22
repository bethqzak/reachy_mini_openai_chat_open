"""OpenAI Realtime API client for Reachy Mini.

Speaks the GA Realtime WebSocket protocol:
  * streams robot microphone audio up as PCM16
  * plays the model's PCM16 speech back through the robot speaker
  * uses server-side VAD for natural turn-taking + barge-in
  * dispatches function calls to robot motion / camera tools
  * injects camera frames as image input for "what do you see?" requests

Runs entirely inside one asyncio event loop (started from a worker thread by
main.py), so all WebSocket sends happen from a single place.
"""

from __future__ import annotations

import asyncio
import base64
import json

import websockets

from .log import get_logger, throttle
from .tools import ToolContext, dispatch, tool_schemas

logger = get_logger("realtime")

REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"

# Event-name variants across API revisions (GA renamed several).
_AUDIO_DELTA = {"response.output_audio.delta", "response.audio.delta"}
_ASSISTANT_TRANSCRIPT = {
    "response.output_audio_transcript.done",
    "response.audio_transcript.done",
}

# High-frequency events we never trace individually at DEBUG.
_NOISY_EVENTS = _AUDIO_DELTA | {
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
    "response.text.delta",
    "response.function_call_arguments.delta",
    "conversation.item.input_audio_transcription.delta",
}


def _connect_error_reason(e: Exception) -> str:
    """Human-readable reason for a failed websocket handshake."""
    # websockets v13+: InvalidStatus with .response.status_code;
    # older versions: InvalidStatusCode with .status_code.
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (401, 403):
        return f"OpenAI rejected the API key ({status})"
    if status is not None:
        return f"OpenAI refused the connection (HTTP {status})"
    return f"could not reach OpenAI ({e.__class__.__name__})"


def _api_error_reason(err: dict) -> str:
    """Human-readable reason from a Realtime API `error` event."""
    code = err.get("code") or err.get("type") or ""
    if code in ("invalid_api_key", "invalid_authentication", "authentication_error"):
        return "OpenAI rejected the API key (401)"
    msg = err.get("message") or "OpenAI returned an error"
    return f"OpenAI rejected the session: {msg}"


class OpenAIRealtimeClient:
    def __init__(self, config, mic_reader, speaker, tool_ctx: ToolContext,
                 logger=None, runtime=None, restart_event=None):
        self.cfg = config
        self.mic = mic_reader
        self.speaker = speaker
        self.ctx = tool_ctx
        self.logger = logger
        # Shared status dict (read by the settings UI) and a live-reconnect flag.
        self.runtime = runtime if runtime is not None else {}
        self.restart_event = restart_event
        self.ws = None
        self._response_active = False
        self._pending_calls: dict[str, str] = {}  # call_id -> name
        self._handled: set[str] = set()  # call_ids already executed

    # ------------------------------------------------------------------ #
    async def run(self, stop_event) -> None:
        url = REALTIME_URL.format(model=self.cfg.model)
        headers = {"Authorization": f"Bearer {self.cfg.openai_api_key}"}
        logger.info("connecting to %s ...", self.cfg.model)

        try:
            ws = await self._connect(url, headers)
        except Exception as e:
            self.runtime["last_error"] = _connect_error_reason(e)
            logger.exception("FAILED to connect (check API key / network / model name)")
            raise
        self.ws = ws
        # Not "connected" yet: the socket can open and still be rejected a
        # moment later (e.g. bad API key). We only report connected once the
        # server accepts the session (session.created in _handle).
        logger.info("websocket open, waiting for session.created ...")

        try:
            await self._configure()
            # Greet only on the very first connection, not on reconnects/restarts.
            if (self.cfg.greet_on_start and self.cfg.greeting
                    and not self.runtime.get("greeted")):
                await self._send({
                    "type": "response.create",
                    "response": {"instructions": self.cfg.greeting},
                })
                self.runtime["greeted"] = True

            tasks = [
                asyncio.create_task(self._mic_loop(stop_event)),
                asyncio.create_task(self._recv_loop()),
            ]
            while not stop_event.is_set() and not self._restart_requested():
                if any(t.done() for t in tasks):
                    for t in tasks:
                        if t.done():
                            logger.debug("%s finished first; ending session",
                                         t.get_coro().__qualname__)
                    break
                await asyncio.sleep(0.2)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.runtime["connected"] = False
            try:
                await ws.close()
            except Exception:
                pass
            logger.info("disconnected.")

    def _restart_requested(self) -> bool:
        return self.restart_event is not None and self.restart_event.is_set()

    async def _connect(self, url, headers):
        # websockets renamed extra_headers -> additional_headers in v13.
        try:
            return await websockets.connect(url, additional_headers=headers, max_size=None)
        except TypeError:
            return await websockets.connect(url, extra_headers=headers, max_size=None)

    # ------------------------------------------------------------------ #
    async def _configure(self) -> None:
        session = {
            "type": "realtime",
            "model": self.cfg.model,
            "output_modalities": ["audio"],
            "instructions": self.cfg.instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.cfg.openai_sample_rate},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 600,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                    "transcription": {"model": "whisper-1",
                                      "language": self.cfg.transcription_language},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": self.cfg.openai_sample_rate},
                    "voice": self.cfg.voice,
                },
            },
        }
        if self.cfg.tools_enabled:
            session["tools"] = tool_schemas(self.cfg.enable_camera)
            session["tool_choice"] = "auto"
        logger.debug("sending session.update (voice=%s, %d tools)",
                     self.cfg.voice, len(session.get("tools", [])))
        await self._send({"type": "session.update", "session": session})

    async def _send(self, event: dict) -> None:
        if self.ws is None:
            return
        await self.ws.send(json.dumps(event))

    # ------------------------------------------------------------------ #
    async def _mic_loop(self, stop_event) -> None:
        loop = asyncio.get_event_loop()
        while not stop_event.is_set():
            if self.cfg.half_duplex:
                self.mic.muted = self.speaker.is_talking
            chunk = await loop.run_in_executor(None, self.mic.read_chunk)
            if chunk:
                try:
                    await self._send({"type": "input_audio_buffer.append", "audio": chunk})
                except Exception as e:
                    logger.warning("mic upload stopped (websocket closed?): %s", e)
                    break
            else:
                await asyncio.sleep(0.005)
        logger.debug("mic loop ended.")

    # ------------------------------------------------------------------ #
    async def _recv_loop(self) -> None:
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                except Exception:
                    logger.warning("ignoring non-JSON message from server")
                    continue
                await self._handle(data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self.runtime.get("connected") and not self.runtime.get("last_error"):
                # Closed before the session was ever accepted.
                rcvd = getattr(e, "rcvd", None)  # close frame, if any
                if rcvd is not None:
                    detail = getattr(rcvd, "reason", "") or f"code {getattr(rcvd, 'code', '?')}"
                    self.runtime["last_error"] = f"OpenAI closed the connection ({detail})"
                else:
                    self.runtime["last_error"] = _connect_error_reason(e)
            logger.warning("recv loop ended: %s", e)

    async def _handle(self, data: dict) -> None:
        etype = data.get("type", "")
        if etype not in _NOISY_EVENTS:
            logger.debug("event: %s", etype)

        if etype == "session.created":
            # Only now has OpenAI actually accepted the session (a bad key can
            # pass the websocket handshake and get rejected afterwards).
            self.runtime["connected"] = True
            self.runtime["last_error"] = None
            logger.info("connected (session accepted by OpenAI).")
            return

        if etype in _AUDIO_DELTA:
            delta = data.get("delta")
            if delta:
                try:
                    self.speaker.enqueue_pcm16(base64.b64decode(delta))
                except Exception as e:
                    if throttle("audio-delta"):
                        logger.warning("could not play audio delta: %s", e)
            return

        if etype == "input_audio_buffer.speech_started":
            # User barged in: drop queued robot speech immediately.
            logger.debug("user speech detected%s",
                         " (interrupting response)" if self._response_active else "")
            self.speaker.clear()
            if self._response_active:
                try:
                    await self._send({"type": "response.cancel"})
                except Exception as e:
                    logger.debug("response.cancel failed: %s", e)
            return

        if etype == "response.created":
            self._response_active = True
            return

        if etype == "response.done":
            self._response_active = False
            resp = data.get("response", {})
            status = resp.get("status")
            if status not in (None, "completed"):
                # e.g. "failed" (quota, server error) or "incomplete"
                logger.warning("response ended with status=%s details=%s",
                               status, json.dumps(resp.get("status_details") or {}))
            # Fallback: pick up any function calls not already handled.
            for item in resp.get("output", []):
                if item.get("type") == "function_call":
                    cid = item.get("call_id")
                    if cid and cid not in self._handled:
                        await self._run_tool(item.get("name"), item.get("arguments", "{}"), cid)
            return

        if etype == "response.output_item.added":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                self._pending_calls[item.get("call_id", "")] = item.get("name", "")
            return

        if etype == "response.function_call_arguments.done":
            cid = data.get("call_id", "")
            name = self._pending_calls.get(cid) or data.get("name")
            await self._run_tool(name, data.get("arguments", "{}"), cid)
            return

        if etype in _ASSISTANT_TRANSCRIPT:
            text = data.get("transcript")
            if text:
                logger.info("robot: %s", text)
                if self.logger:
                    self.logger.log_assistant(text)
            return

        if etype == "conversation.item.input_audio_transcription.completed":
            text = data.get("transcript")
            if text:
                logger.info("user: %s", text.strip())
                if self.logger:
                    self.logger.log_user(text)
            return

        if etype == "error":
            err = data.get("error", data) or {}
            logger.error("API error: %s", json.dumps(err))
            if not self.runtime.get("connected"):
                # Rejected before the session was accepted — keep the reason
                # so the settings UI can say why instead of "reconnecting…".
                self.runtime["last_error"] = _api_error_reason(err)
            return

    # ------------------------------------------------------------------ #
    async def _run_tool(self, name: str, arguments: str, call_id: str) -> None:
        if not call_id or call_id in self._handled:
            return
        self._handled.add(call_id)
        try:
            args = json.loads(arguments) if arguments else {}
        except Exception:
            logger.warning("tool %s: could not parse arguments %r", name, arguments)
            args = {}
        try:
            output, image_uri = dispatch(name, args, self.ctx)
        except Exception:
            logger.exception("tool %s crashed", name)
            output, image_uri = ('{"ok": false, "error": "tool crashed"}', None)

        logger.info("tool %s(%s) -> %s", name, args, output)
        if self.logger:
            self.logger.log_tool(name, args, output, image_data_uri=image_uri)

        # Return the function result to the model.
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id, "output": output},
        })
        # If the tool produced an image, let the model actually see it.
        if image_uri:
            await self._send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": image_uri}],
                },
            })
        # Continue the turn.
        try:
            await self._send({"type": "response.create"})
        except Exception as e:
            logger.warning("could not continue turn after tool %s: %s", name, e)
