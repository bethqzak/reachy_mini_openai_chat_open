"""Camera capture and lightweight on-device vision for Reachy Mini.

Two distinct capabilities live here:

  * capture_jpeg_data_uri(): grab a frame and encode it so it can be sent to
    the OpenAI Realtime model as an `input_image` (this is the "what do you
    see?" image-recognition feature — the heavy lifting is done by the model).

  * FaceTracker: a small OpenCV Haar-cascade detector that runs locally so the
    robot can *follow a face* with its head as ambient behaviour, independent
    of the language model.
"""

from __future__ import annotations

import base64
import threading
import time

import numpy as np

from .log import get_logger, throttle

logger = get_logger("vision")

try:
    import cv2  # type: ignore

    _HAVE_CV2 = True
except Exception as _cv2_err:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAVE_CV2 = False
    logger.warning("OpenCV (cv2) not available: %s — camera vision and "
                   "face tracking disabled", _cv2_err)


# -- face blurring (privacy) ------------------------------------------- #
_privacy_cascades: list | None = None


def _load_privacy_cascades() -> list:
    """Haar cascades used to find faces so they can be pixelated before an
    image leaves the robot. Loaded once, lazily."""
    global _privacy_cascades
    if _privacy_cascades is None:
        _privacy_cascades = []
        if _HAVE_CV2:
            for name in ("haarcascade_frontalface_default.xml",
                         "haarcascade_profileface.xml"):
                try:
                    c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
                    if not c.empty():
                        _privacy_cascades.append((name, c))
                except Exception:
                    logger.exception("could not load cascade %s", name)
    return _privacy_cascades


def pixelate_faces_bgr(bgr: np.ndarray) -> int:
    """Pixelate every detected face in the BGR frame, in place.

    Uses frontal + profile Haar cascades (profile is also run on the mirrored
    frame, since that cascade only detects one facing direction). Detection is
    not perfect — strongly angled or partly hidden faces can be missed — but
    it covers the common case of people looking at or near the robot.
    Returns the number of regions pixelated.
    """
    cascades = _load_privacy_cascades()
    if not cascades:
        return 0
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes: list[tuple[int, int, int, int]] = []
    for name, cascade in cascades:
        views = [(gray, False)]
        if "profile" in name:
            views.append((cv2.flip(gray, 1), True))
        for view, mirrored in views:
            for (x, y, fw, fh) in cascade.detectMultiScale(
                    view, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)):
                if mirrored:
                    x = w - x - fw
                boxes.append((int(x), int(y), int(fw), int(fh)))
    count = 0
    for x, y, fw, fh in boxes:
        # Grow the box a little so hairline and chin aren't left identifiable.
        mx, my = int(fw * 0.2), int(fh * 0.2)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = bgr[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 16), max(1, (y1 - y0) // 16)),
                           interpolation=cv2.INTER_LINEAR)
        bgr[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0),
                                       interpolation=cv2.INTER_NEAREST)
        count += 1
    return count


def capture_jpeg(media, max_width: int = 768, quality: int = 70,
                 blur_faces: bool = True, quiet: bool = False) -> bytes | None:
    """Grab one camera frame and return it as JPEG bytes.

    With `blur_faces`, detected faces are pixelated before the frame is
    encoded. `quiet` rate-limits the failure logging (for callers that poll,
    such as the settings page's live view, so a dead camera doesn't flood the
    log at frame rate). Returns None if no frame or if OpenCV is unavailable.
    """
    def warn(key: str, msg: str, *args) -> None:
        if not quiet or throttle("capture-" + key):
            logger.warning(msg, *args)

    if not _HAVE_CV2:
        warn("cv2", "capture failed: OpenCV not installed")
        return None
    try:
        frame = media.get_frame()
    except Exception as e:
        if quiet:
            warn("raised", "capture failed: media.get_frame() raised: %s", e)
        else:
            logger.exception("capture failed: media.get_frame() raised")
        return None
    if frame is None:
        warn("none", "capture failed: camera returned no frame")
        return None
    frame = np.asarray(frame)
    if frame.ndim != 3:
        warn("shape", "capture failed: unexpected frame shape %s", frame.shape)
        return None

    # Downscale to keep the payload small / fast.
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    # The reachy_mini SDK hands out BGR frames (see MediaManager.get_frame),
    # which is also what OpenCV encodes — no channel swap needed.
    bgr = np.ascontiguousarray(frame)
    if blur_faces:
        # Fail closed: if we can't blur, we don't send the image at all.
        if not _load_privacy_cascades():
            warn("cascade", "capture blocked: face blurring requested but no "
                 "face detector is available")
            return None
        n = pixelate_faces_bgr(bgr)
        if n:
            logger.info("pixelated %d face(s) before encoding", n)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        warn("encode", "capture failed: JPEG encoding error")
        return None
    data = buf.tobytes()
    logger.debug("captured %dx%d frame (%d KB as JPEG)", w, h, len(data) // 1024)
    return data


def capture_jpeg_data_uri(media, max_width: int = 768, quality: int = 70,
                          blur_faces: bool = True) -> str | None:
    """Grab one camera frame and return a `data:image/jpeg;base64,...` URI.

    With `blur_faces` (the default), detected faces are pixelated before the
    frame is encoded, so no unblurred image ever leaves the robot.
    Returns None if no frame or if OpenCV is unavailable.
    """
    data = capture_jpeg(media, max_width=max_width, quality=quality,
                        blur_faces=blur_faces)
    if data is None:
        return None
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class FaceTracker(threading.Thread):
    """Detects the largest face and exposes a normalized offset from center.

    offset_x / offset_y are in roughly [-1, 1]; the MotionController turns them
    into head yaw/pitch so the robot looks toward the person. Only runs while
    `enabled` is set, so it costs nothing when face tracking is off.
    """

    def __init__(self, media, rate_hz: float = 8.0):
        super().__init__(name="FaceTracker", daemon=True)
        self.media = media
        self.period = 1.0 / rate_hz
        self._stop = threading.Event()
        self.enabled = False

        self._lock = threading.Lock()
        self._offset = (0.0, 0.0)
        self._has_face = False
        self._last_seen = 0.0

        self._cascade = None
        if _HAVE_CV2:
            try:
                path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(path)
                if self._cascade.empty():
                    logger.warning("face cascade file empty/missing at %s", path)
                    self._cascade = None
            except Exception:
                logger.exception("could not load face cascade")
                self._cascade = None

    @property
    def available(self) -> bool:
        return self._cascade is not None

    def set_enabled(self, value: bool) -> None:
        if bool(value) != self.enabled:
            logger.info("face tracking %s", "enabled" if value else "disabled")
        self.enabled = bool(value)
        if not value:
            with self._lock:
                self._offset = (0.0, 0.0)
                self._has_face = False

    def get_offset(self) -> tuple[float, float, bool]:
        with self._lock:
            fresh = self._has_face and (time.monotonic() - self._last_seen) < 1.0
            return self._offset[0], self._offset[1], fresh

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if self._cascade is None:
            return
        while not self._stop.is_set():
            if not self.enabled:
                time.sleep(0.1)
                continue
            try:
                frame = self.media.get_frame()
            except Exception as e:
                if throttle("face-frame"):
                    logger.warning("face tracker could not read camera: %s", e)
                frame = None
            if frame is not None:
                self._process(np.asarray(frame))
            time.sleep(self.period)

    def _process(self, frame: np.ndarray) -> None:
        if frame.ndim != 3:
            return
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # SDK frames are BGR
        faces = self._cascade.detectMultiScale(gray, 1.2, 5, minSize=(48, 48))
        if len(faces) == 0:
            with self._lock:
                self._has_face = False
            return
        # Largest face wins.
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        # Normalize: face right of center -> positive offset_x.
        off_x = (cx - w / 2.0) / (w / 2.0)
        off_y = (cy - h / 2.0) / (h / 2.0)
        with self._lock:
            self._offset = (float(np.clip(off_x, -1, 1)), float(np.clip(off_y, -1, 1)))
            self._has_face = True
            self._last_seen = time.monotonic()
