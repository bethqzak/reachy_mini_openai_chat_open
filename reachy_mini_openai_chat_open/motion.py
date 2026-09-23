"""Motion controller for Reachy Mini.

A single thread streams targets to the robot at ~50 Hz, blending several
layers so the robot always feels alive:

  * ambient  : slow "breathing" bob + tiny idle drift (always on)
  * talking  : antenna + head wobble driven by live speech energy
  * tracking  : head turns toward a detected face (when enabled)
  * gestures : one-shot expressive moves (nod, shake, wiggle, emotions, dance)
               triggered by the language model via the `express` tool

Every layer produces small additive offsets in a common unit set
(degrees for angles, mm for height, radians for body_yaw / antennas) which are
summed, clamped to safe ranges, and sent with `set_target`.
"""

from __future__ import annotations

import inspect
import math
import threading
import time

import numpy as np

from .log import get_logger, throttle, throttle_skipped

logger = get_logger("motion")

try:
    from reachy_mini.utils import create_head_pose  # type: ignore
except Exception:  # allows import / syntax-check off-robot
    create_head_pose = None  # type: ignore
    logger.warning("reachy_mini.utils.create_head_pose unavailable; "
                   "head pose control disabled")

# The SDK's own neutral antenna angles (~10° apart to avoid shaking when
# perfectly vertical). Fall back to the same literal if the import fails.
try:
    from reachy_mini.reachy_mini import (  # type: ignore
        INIT_ANTENNAS_JOINT_POSITIONS as _NEUTRAL_ANTENNAS,
    )
except Exception:
    _NEUTRAL_ANTENNAS = [-0.1745, 0.1745]


# --------------------------------------------------------------------------- #
# Gesture library — each returns an offset dict given local time t (seconds).
# Envelopes start and end at ~0 so gestures blend in and out without jumps.
# --------------------------------------------------------------------------- #
def _env(t: float, dur: float) -> float:
    if t <= 0 or t >= dur:
        return 0.0
    return math.sin(math.pi * t / dur)


def _zero() -> dict:
    return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "z": 0.0, "body": 0.0,
            "ant_l": 0.0, "ant_r": 0.0}


def _gesture(name: str):
    """Return (duration_seconds, fn(t)->offset_dict) for a named gesture."""
    name = (name or "").lower().strip()

    def nod_yes(t):
        o = _zero()
        o["pitch"] = -14.0 * math.sin(2 * math.pi * 1.6 * t) * _env(t, 1.3)
        return o

    def shake_no(t):
        o = _zero()
        o["yaw"] = 22.0 * math.sin(2 * math.pi * 1.7 * t) * _env(t, 1.3)
        return o

    def tilt(t):
        o = _zero()
        e = _env(t, 1.6)
        o["roll"] = 16.0 * e
        o["ant_l"] = 0.35 * e
        o["ant_r"] = -0.15 * e
        return o

    def look_around(t):
        o = _zero()
        o["yaw"] = 28.0 * math.sin(2 * math.pi * 0.45 * t) * _env(t, 2.6)
        return o

    def wiggle(t):
        o = _zero()
        e = _env(t, 1.2)
        w = 0.5 * math.sin(2 * math.pi * 4.0 * t) * e
        o["ant_l"] = w
        o["ant_r"] = -w
        return o

    def happy(t):
        o = _zero()
        e = _env(t, 1.6)
        o["ant_l"] = 0.45 * e
        o["ant_r"] = 0.45 * e
        o["z"] = 8.0 * abs(math.sin(2 * math.pi * 1.5 * t)) * e
        o["yaw"] = 6.0 * math.sin(2 * math.pi * 2.0 * t) * e
        return o

    def excited(t):
        o = _zero()
        e = _env(t, 1.8)
        w = 0.55 * math.sin(2 * math.pi * 5.0 * t) * e
        o["ant_l"] = 0.3 * e + w
        o["ant_r"] = 0.3 * e - w
        o["z"] = 9.0 * abs(math.sin(2 * math.pi * 2.5 * t)) * e
        o["pitch"] = -5.0 * e
        return o

    def sad(t):
        o = _zero()
        e = _env(t, 2.0)
        o["pitch"] = 14.0 * e
        o["ant_l"] = -0.4 * e
        o["ant_r"] = -0.4 * e
        o["z"] = -6.0 * e
        return o

    def surprised(t):
        o = _zero()
        e = _env(t, 1.1)
        o["pitch"] = -16.0 * e
        o["z"] = 10.0 * e
        o["ant_l"] = 0.6 * e
        o["ant_r"] = 0.6 * e
        return o

    def sleepy(t):
        o = _zero()
        e = _env(t, 2.6)
        o["pitch"] = 16.0 * e
        o["roll"] = 8.0 * math.sin(2 * math.pi * 0.3 * t) * e
        o["ant_l"] = -0.5 * e
        o["ant_r"] = -0.5 * e
        return o

    def dance(t):
        o = _zero()
        e = _env(t, 4.2)
        o["yaw"] = 18.0 * math.sin(2 * math.pi * 0.8 * t) * e
        o["body"] = 0.28 * math.sin(2 * math.pi * 0.8 * t + math.pi / 2) * e
        o["roll"] = 10.0 * math.sin(2 * math.pi * 1.6 * t) * e
        o["z"] = 7.0 * math.sin(2 * math.pi * 2.4 * t) * e
        w = 0.4 * math.sin(2 * math.pi * 3.0 * t) * e
        o["ant_l"] = w
        o["ant_r"] = -w
        return o

    table = {
        "nod_yes": (1.3, nod_yes),
        "yes": (1.3, nod_yes),
        "nod": (1.3, nod_yes),
        "shake_no": (1.3, shake_no),
        "no": (1.3, shake_no),
        "tilt_head": (1.6, tilt),
        "curious": (1.6, tilt),
        "look_around": (2.6, look_around),
        "wiggle_antennas": (1.2, wiggle),
        "happy": (1.6, happy),
        "excited": (1.8, excited),
        "sad": (2.0, sad),
        "surprised": (1.1, surprised),
        "confused": (1.6, tilt),
        "sleepy": (2.6, sleepy),
        "dance": (4.2, dance),
    }
    return table.get(name)


AVAILABLE_EXPRESSIONS = [
    "nod_yes", "shake_no", "tilt_head", "look_around", "wiggle_antennas",
    "happy", "excited", "sad", "surprised", "confused", "sleepy", "dance",
]


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class MotionController(threading.Thread):
    def __init__(self, reachy_mini, config, speaker=None, face_tracker=None):
        super().__init__(name="MotionController", daemon=True)
        self.mini = reachy_mini
        self.cfg = config
        self.speaker = speaker
        self.face_tracker = face_tracker
        self._stop_event = threading.Event()

        self._gestures: list[tuple[float, float, callable]] = []  # (start, dur, fn)
        self._glock = threading.Lock()

        # smoothed tracking state
        self._track_yaw = 0.0
        self._track_pitch = 0.0

        # Detect create_head_pose kwargs once (degrees / mm support varies).
        self._pose_kwargs = {}
        if create_head_pose is not None:
            try:
                params = inspect.signature(create_head_pose).parameters
                if "degrees" in params:
                    self._pose_kwargs["degrees"] = True
                if "mm" in params:
                    self._pose_kwargs["mm"] = True
            except (ValueError, TypeError):
                pass

    # -- public API (thread-safe) -----------------------------------------
    def trigger(self, name: str) -> bool:
        """Queue an expressive gesture. Returns False for unknown names."""
        if name and name.lower().strip() == "reset":
            logger.debug("gesture reset")
            with self._glock:
                self._gestures.clear()
            return True
        g = _gesture(name)
        if g is None:
            logger.warning("unknown gesture %r", name)
            return False
        dur, fn = g
        logger.debug("gesture %r queued (%.1fs)", name, dur)
        with self._glock:
            self._gestures.append((time.monotonic(), dur, fn))
        return True

    def stop(self) -> None:
        self._stop_event.set()

    def go_neutral(self, duration: float = 1.0) -> None:
        """Glide the head, antennas and body back to the neutral (init) pose.

        Call after :meth:`stop` has been issued and the loop has exited, so
        the 50 Hz set_target stream is no longer fighting the move. Used on
        shutdown so the robot is left upright and centred, not asleep.
        """
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=1.0)
        neutral = self._make_head(0.0, 0.0, 0.0, 0.0)
        antennas = np.array(_NEUTRAL_ANTENNAS)
        try:
            self.mini.goto_target(head=neutral, antennas=antennas,
                                  duration=duration, body_yaw=0.0)
        except Exception as e:
            logger.warning("goto_target to neutral failed (%s); "
                           "falling back to set_target", e)
            try:
                kwargs = {"antennas": antennas, "body_yaw": 0.0}
                if neutral is not None:
                    kwargs["head"] = neutral
                self.mini.set_target(**kwargs)
            except Exception as e2:
                logger.warning("could not reset to neutral pose: %s", e2)

    # -- internals ---------------------------------------------------------
    def _active_gesture_offsets(self, now: float) -> dict:
        total = _zero()
        keep = []
        with self._glock:
            for start, dur, fn in self._gestures:
                t = now - start
                if t >= dur:
                    continue
                keep.append((start, dur, fn))
                off = fn(t)
                for k in total:
                    total[k] += off.get(k, 0.0)
            self._gestures = keep
        return total

    def _make_head(self, yaw, pitch, roll, z_mm):
        if create_head_pose is None:
            return None
        kw = dict(self._pose_kwargs)
        if "degrees" in kw:
            angles = dict(yaw=yaw, pitch=pitch, roll=roll)
        else:
            angles = dict(yaw=math.radians(yaw), pitch=math.radians(pitch),
                          roll=math.radians(roll))
        if "mm" in kw:
            z = z_mm
        else:
            z = z_mm / 1000.0
        try:
            return create_head_pose(z=z, **angles, **kw)
        except TypeError:
            # Fallback: minimal signature.
            return create_head_pose(yaw=angles["yaw"], pitch=angles["pitch"],
                                    roll=angles["roll"])

    def run(self) -> None:
        t0 = time.monotonic()
        dt = 0.02  # 50 Hz
        while not self._stop_event.is_set():
            now = time.monotonic()
            t = now - t0

            yaw = pitch = roll = zmm = body = 0.0
            ant_l = ant_r = 0.0

            # 1) ambient breathing
            if self.cfg.ambient_enabled:
                zmm += 3.5 * math.sin(2 * math.pi * 0.22 * t)
                pitch += 1.5 * math.sin(2 * math.pi * 0.22 * t + 0.6)
                yaw += 2.5 * math.sin(2 * math.pi * 0.05 * t)
                ant_l += 0.05 * math.sin(2 * math.pi * 0.18 * t)
                ant_r += 0.05 * math.sin(2 * math.pi * 0.18 * t + 0.3)

            # 2) talking wobble
            energy = self.speaker.talking_energy if self.speaker else 0.0
            if energy > 0.01:
                w = math.sin(2 * math.pi * 6.5 * t)
                ant_l += 0.30 * energy * w
                ant_r += 0.30 * energy * -w
                pitch += 2.5 * energy * math.sin(2 * math.pi * 3.0 * t)
                yaw += 2.0 * energy * math.sin(2 * math.pi * 2.0 * t)

            # 3) face tracking
            if self.face_tracker is not None and self.face_tracker.enabled:
                ox, oy, fresh = self.face_tracker.get_offset()
                tgt_yaw = -22.0 * ox if fresh else 0.0
                tgt_pitch = 16.0 * oy if fresh else 0.0
                self._track_yaw += 0.15 * (tgt_yaw - self._track_yaw)
                self._track_pitch += 0.15 * (tgt_pitch - self._track_pitch)
                yaw += self._track_yaw
                pitch += self._track_pitch

            # 4) gestures
            g = self._active_gesture_offsets(now)
            yaw += g["yaw"]
            pitch += g["pitch"]
            roll += g["roll"]
            zmm += g["z"]
            body += g["body"]
            ant_l += g["ant_l"]
            ant_r += g["ant_r"]

            # clamp to safe ranges
            yaw = float(np.clip(yaw, -40, 40))
            pitch = float(np.clip(pitch, -25, 28))
            roll = float(np.clip(roll, -25, 25))
            zmm = float(np.clip(zmm, -15, 15))
            body = float(np.clip(body, -0.5, 0.5))
            ant_l = float(np.clip(ant_l, -0.9, 0.9))
            ant_r = float(np.clip(ant_r, -0.9, 0.9))

            try:
                head = self._make_head(yaw, pitch, roll, zmm)
                kwargs = {"antennas": np.array([ant_l, ant_r]), "body_yaw": body}
                if head is not None:
                    kwargs["head"] = head
                self.mini.set_target(**kwargs)
            except Exception as e:
                # At 50 Hz this would flood the log; report once per 10s.
                if throttle("set-target"):
                    logger.warning("set_target failed (+%d suppressed): %s",
                                   throttle_skipped("set-target"), e)

            time.sleep(dt)
        logger.debug("motion loop stopped.")
