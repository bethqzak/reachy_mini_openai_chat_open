"""Audio plumbing between the Reachy Mini media backend and OpenAI Realtime.

Robot side  : float32, mono/stereo, 16 kHz (via reachy_mini.media)
OpenAI side : base64 PCM16, mono, 24 kHz

This module owns:
  * conversion helpers (float32 <-> PCM16, resampling)
  * a MicReader that yields base64 PCM16 chunks ready to send to OpenAI
  * a SpeakerPlayer thread that turns incoming PCM16 chunks back into robot
    audio, exposes a live "talking energy" level for speech-reactive motion,
    and can be cleared instantly for barge-in.
"""

from __future__ import annotations

import base64
import queue
import threading
import time

import numpy as np
from scipy.signal import resample_poly

from .log import get_logger, throttle, throttle_skipped

logger = get_logger("audio")


# --------------------------------------------------------------------------- #
# Conversion helpers
# --------------------------------------------------------------------------- #
def float_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """float32 in [-1, 1] -> little-endian PCM16 bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_bytes_to_float(data: bytes) -> np.ndarray:
    """PCM16 bytes -> float32 in [-1, 1]."""
    ints = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return ints / 32768.0


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    g = np.gcd(int(src_rate), int(dst_rate))
    up = dst_rate // g
    down = src_rate // g
    return resample_poly(samples, up, down).astype(np.float32)


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2 and samples.shape[1] > 1:
        return samples.mean(axis=1)
    return samples.reshape(-1)


# Below this level the limiter is a no-op, so normal-volume audio is passed
# through bit-for-bit; above it the curve bends smoothly towards 1.0.
LIMIT_KNEE = 0.8


def soft_limit(samples: np.ndarray, knee: float = LIMIT_KNEE) -> np.ndarray:
    """Bound samples to [-1, 1] with a soft knee rather than a hard clip.

    Boosting past 100% has to go somewhere. `np.clip` would square off the
    peaks — harsh distortion, and a square wave is the worst thing you can ask
    a small speaker to reproduce. Instead the curve stays linear (unity gain)
    below `knee` and compresses asymptotically above it, so loud passages get
    louder and rounder instead of clipped, and the output never leaves [-1, 1].
    """
    over = np.abs(samples) > knee
    if not over.any():
        return samples
    out = samples.copy()
    span = 1.0 - knee
    excess = (np.abs(samples[over]) - knee) / span
    out[over] = np.sign(samples[over]) * (knee + span * np.tanh(excess))
    return out


# --------------------------------------------------------------------------- #
# Microphone
# --------------------------------------------------------------------------- #
class MicReader:
    """Pulls audio from the robot mic and produces base64 PCM16 @ 24 kHz."""

    def __init__(self, media, robot_rate: int, openai_rate: int = 24000):
        self.media = media
        self.robot_rate = robot_rate
        self.openai_rate = openai_rate
        self.muted = False  # set True during half-duplex playback

    def read_chunk(self) -> str | None:
        """Return a base64 PCM16 chunk, or None if no audio is available.

        This call is blocking-ish (delegated to an executor by the caller).
        """
        try:
            samples = self.media.get_audio_sample()
        except Exception as e:
            if throttle("mic-read"):
                logger.warning("mic read failed (+%d suppressed): %s",
                               throttle_skipped("mic-read"), e)
            return None
        if samples is None:
            return None
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return None
        if self.muted:
            # Still drain the buffer, but send silence so timing stays sane.
            samples = np.zeros_like(samples)
        mono = to_mono(samples)
        up = resample(mono, self.robot_rate, self.openai_rate)
        return base64.b64encode(float_to_pcm16_bytes(up)).decode("ascii")


# --------------------------------------------------------------------------- #
# Speaker
# --------------------------------------------------------------------------- #
class SpeakerPlayer(threading.Thread):
    """Plays OpenAI PCM16 audio through the robot speaker in small chunks.

    The model streams a reply far faster than it can be spoken, and the
    daemon's playback pipeline accepts audio without back-pressure — so a
    naive "push as it arrives" player hands the whole reply to the daemon
    within a second or two and then knows nothing about it: it can't say
    whether the robot is still talking, how far it has got, or stop it.

    This player therefore paces itself to real time, keeping only `lead_s`
    of audio inside the daemon. That keeps everything else truthful:

      * `is_talking` / `talking_energy` follow the voice, not the download,
      * `played_ms` says how much of the current reply has actually been
        heard (for `conversation.item.truncate` after a barge-in), and
      * `clear()` really stops the sound — it drops the local queue *and*
        flushes the little that's already in the daemon.

    Audio is tracked per server item (`item_id`): one reply's tail can still
    be playing when the next reply starts, and a barge-in has to truncate
    the right one.
    """

    # Audio chunk pushed to the daemon per step. Small so a barge-in cut is
    # tight and the energy signal is responsive.
    CHUNK_S = 0.04

    def __init__(self, media, robot_rate: int, openai_rate: int = 24000,
                 lead_s: float = 0.25):
        super().__init__(name="SpeakerPlayer", daemon=True)
        self.media = media
        self.robot_rate = robot_rate
        self.openai_rate = openai_rate
        # How far ahead of real time the daemon is fed. Too small and a busy
        # host stutters; too large and that much audio survives a barge-in.
        self.lead_s = max(0.05, float(lead_s))
        self._q: "queue.Queue[tuple[int, str | None, np.ndarray] | None]" = queue.Queue()
        self._stop = threading.Event()
        self._energy = 0.0
        self._energy_lock = threading.Lock()
        self._playing = threading.Event()
        # Output volume multiplier (1.0 = unchanged). Plain attribute, no lock:
        # a float assignment is atomic, so the playback thread can never see a
        # torn value — worst case it uses the old one for one more chunk.
        self.volume = 1.0
        # Playback bookkeeping for the utterance being spoken, under _plock.
        # `gen` is bumped by clear(): a chunk dequeued under an older
        # generation is dropped instead of being pushed after the flush.
        self._plock = threading.Lock()
        self._gen = 0
        self._item: str | None = None      # server item id of the utterance
        self._anchor: float | None = None  # monotonic time its playback began
        self._pushed_s = 0.0               # seconds of it handed to the daemon
        self._flush = self._find_flush(media)

    # -- daemon flush --------------------------------------------------------
    @staticmethod
    def _find_flush(media):
        """Locate the daemon's playback flush, if this backend has one.

        reachy_mini >= 1.8.4 exposes `clear_player()` on the audio backend
        (`media.audio`); newer MediaManagers may forward it directly. Absent
        on fakes/mocks, in which case clear() only drops the local queue.
        """
        for holder in (media, getattr(media, "audio", None)):
            fn = getattr(holder, "clear_player", None)
            if callable(fn):
                return fn
        logger.info("media backend has no clear_player(); barge-in will only "
                    "drop audio not yet handed to the daemon")
        return None

    # -- producer side -----------------------------------------------------
    def enqueue_pcm16(self, data: bytes, item_id: str | None = None) -> None:
        mono = pcm16_bytes_to_float(data)
        down = resample(mono, self.openai_rate, self.robot_rate)
        chunk = max(1, int(self.robot_rate * self.CHUNK_S))
        with self._plock:
            gen = self._gen
        for i in range(0, len(down), chunk):
            self._q.put((gen, item_id, down[i : i + chunk]))

    def clear(self) -> tuple[str | None, float]:
        """Stop the robot talking now (barge-in / interruption).

        Drops everything queued locally and flushes what the daemon already
        holds. Returns `(item_id, heard_ms)` for the utterance that was
        playing — the figure to truncate the server-side item to — or
        `(None, 0.0)` if the speaker was idle.
        """
        with self._plock:
            if self._playing.is_set():
                item, heard = self._item, self._played_ms_locked()
            else:
                item, heard = None, 0.0
            self._gen += 1
            self._item = None
            self._anchor = None
            self._pushed_s = 0.0
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._flush is not None:
            try:
                self._flush()
            except Exception as e:
                if throttle("speaker-flush"):
                    logger.warning("daemon flush failed (+%d suppressed): %s",
                                   throttle_skipped("speaker-flush"), e)
        with self._energy_lock:
            self._energy = 0.0
        self._playing.clear()
        return item, heard

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)

    # -- consumers -----------------------------------------------------------
    @property
    def talking_energy(self) -> float:
        with self._energy_lock:
            return self._energy

    @property
    def is_talking(self) -> bool:
        return self._playing.is_set()

    @property
    def played_ms(self) -> float:
        """Milliseconds of the current utterance the speaker has rendered."""
        with self._plock:
            return self._played_ms_locked()

    def _played_ms_locked(self) -> float:
        if self._anchor is None:
            return 0.0
        elapsed = time.monotonic() - self._anchor
        return 1000.0 * max(0.0, min(elapsed, self._pushed_s))

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                entry = self._q.get(timeout=0.05)
            except queue.Empty:
                self._on_idle()
                continue
            if entry is None:
                break
            gen, item_id, chunk = entry
            with self._plock:
                if gen != self._gen:
                    continue  # cleared while this chunk was in flight
                now = time.monotonic()
                if item_id != self._item or self._anchor is None:
                    # A new utterance. It starts when the previous one's tail
                    # (still inside the daemon) has finished playing.
                    prev_end = (self._anchor + self._pushed_s
                                if self._anchor is not None else now)
                    self._item = item_id
                    self._anchor = max(now, prev_end)
                    self._pushed_s = 0.0
                ahead = self._pushed_s - (now - self._anchor)
                if ahead < 0.0:
                    # Underrun (model stalled, host hiccup): the daemon ran
                    # dry, so playback resumes *now* — re-anchor rather than
                    # trying to catch up by dumping everything at once.
                    self._anchor = now - self._pushed_s
                    ahead = 0.0
                self._pushed_s += len(chunk) / self.robot_rate
            if ahead > self.lead_s:
                time.sleep(ahead - self.lead_s)
                with self._plock:
                    if gen != self._gen:
                        continue
            self._playing.set()
            rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
            with self._energy_lock:
                # smooth, and scale so normal speech lands near ~1.0
                target = min(1.0, rms * 6.0)
                self._energy = 0.6 * self._energy + 0.4 * target
            # Energy above is measured on the model's signal so speech-reactive
            # motion stays consistent regardless of playback volume; only the
            # audio actually pushed to the speaker is scaled.
            vol = self.volume
            out = chunk if vol == 1.0 else soft_limit(chunk * vol)
            try:
                self.media.push_audio_sample(out.reshape(-1, 1))
            except Exception as e:
                if throttle("speaker-push"):
                    logger.warning("speaker push failed (+%d suppressed): %s",
                                   throttle_skipped("speaker-push"), e)

    def _on_idle(self) -> None:
        """Queue empty: the robot is only silent once the daemon's tail is out.

        Up to `lead_s` of audio is still inside the daemon after our last
        push, so `is_talking` holds until the wall clock catches up with
        what was pushed. The utterance bookkeeping is kept — a stall between
        two audio deltas must not restart the played-ms count.
        """
        with self._plock:
            if self._anchor is None:
                return
            if time.monotonic() - self._anchor < self._pushed_s:
                return
        self._playing.clear()
        with self._energy_lock:
            self._energy = 0.0

