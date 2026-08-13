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

    Running in its own thread lets us clear the queue instantly for barge-in,
    which the buffered media backend alone would not allow.
    """

    def __init__(self, media, robot_rate: int, openai_rate: int = 24000):
        super().__init__(name="SpeakerPlayer", daemon=True)
        self.media = media
        self.robot_rate = robot_rate
        self.openai_rate = openai_rate
        self._q: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._stop = threading.Event()
        self._energy = 0.0
        self._energy_lock = threading.Lock()
        self._playing = threading.Event()
        # Output volume multiplier (1.0 = unchanged). Plain attribute, no lock:
        # a float assignment is atomic, so the playback thread can never see a
        # torn value — worst case it uses the old one for one more chunk.
        self.volume = 1.0

    # -- producer side -----------------------------------------------------
    def enqueue_pcm16(self, data: bytes) -> None:
        mono = pcm16_bytes_to_float(data)
        down = resample(mono, self.openai_rate, self.robot_rate)
        # Chop into ~40 ms chunks so barge-in clears quickly.
        chunk = max(1, int(self.robot_rate * 0.04))
        for i in range(0, len(down), chunk):
            self._q.put(down[i : i + chunk])

    def clear(self) -> None:
        """Drop everything queued but not yet pushed (barge-in / interruption)."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        with self._energy_lock:
            self._energy = 0.0

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)

    # -- consumer for motion -----------------------------------------------
    @property
    def talking_energy(self) -> float:
        with self._energy_lock:
            return self._energy

    @property
    def is_talking(self) -> bool:
        return self._playing.is_set()

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        idle_ticks = 0
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.05)
            except queue.Empty:
                idle_ticks += 1
                if idle_ticks > 2:
                    self._playing.clear()
                    with self._energy_lock:
                        self._energy = 0.0
                continue
            if chunk is None:
                break
            idle_ticks = 0
            self._playing.set()
            rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
            with self._energy_lock:
                # smooth, and scale so normal speech lands near ~1.0
                target = min(1.0, rms * 6.0)
                self._energy = 0.6 * self._energy + 0.4 * target
            # Energy above is measured on the model's own signal, so
            # speech-reactive head motion stays the same however loud the
            # speaker is set; only what goes to the hardware is scaled.
            vol = self.volume
            out = chunk if vol == 1.0 else soft_limit(chunk * vol)
            try:
                self.media.push_audio_sample(out.reshape(-1, 1))
            except Exception as e:
                if throttle("speaker-push"):
                    logger.warning("speaker push failed (+%d suppressed): %s",
                                   throttle_skipped("speaker-push"), e)
