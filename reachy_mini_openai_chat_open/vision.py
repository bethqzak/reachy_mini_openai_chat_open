"""Camera capture and on-device vision for Reachy Mini.

Detection and blurring live in their own modules; everything here is two steps:

    faces = detect_faces(bgr)
    blur_faces(bgr, faces)

  * capture_jpeg_data_uri(): grab a frame, blur every face, encode it for the
    OpenAI model. Fails closed, so no unblurred frame ever leaves the robot.
  * FaceTracker: a thread reporting where the nearest face is, for head tracking.
"""

from __future__ import annotations

import base64
import threading
import time

import numpy as np

from .blur import DEFAULT_STYLE, blur_faces
from .detection import (DEFAULT_METHOD, DEFAULT_RUNTIME, available,
                        detect_faces)
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
def pixelate_faces_bgr(bgr: np.ndarray, method: str = DEFAULT_METHOD,
                       runtime: str = DEFAULT_RUNTIME,
                       style: str = DEFAULT_STYLE) -> int | None:
    """Obscure every face in the BGR frame.
    Returns how many were obscured, or None if no detector could run"""
    faces = detect_faces(bgr, method, runtime)
    return None if faces is None else blur_faces(bgr, faces, style)


def capture_jpeg(media, max_width: int = 768, quality: int = 70,
                 blur: bool = True, quiet: bool = False,
                 method: str = DEFAULT_METHOD, runtime: str = DEFAULT_RUNTIME,
                 style: str = DEFAULT_STYLE) -> bytes | None:
    """Grab one camera frame and return it as JPEG bytes.

    With `blur`, faces found by `method` on `runtime` are obscured in `style`
    before the frame is encoded. `quiet` rate-limits the failure logging (for callers that poll,
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
        frame = cv2.resize(frame, (max_width, int(h * max_width / float(w))))

    # get_frame() returns BGR, which is what OpenCV wants throughout.
    bgr = np.ascontiguousarray(frame)
    if blur:
        # Fail closed: if we can't blur, we don't send the image at all.
        n = pixelate_faces_bgr(bgr, method, runtime, style)
        if n is None:
            warn("none", "capture blocked: no face detector is available")
            return None
        if n:
            logger.info("obscured %d face(s) before encoding", n)

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        warn("encode", "capture failed: JPEG encoding error")
        return None
    data = buf.tobytes()
    logger.debug("captured %dx%d frame (%d KB as JPEG)", w, h, len(data) // 1024)
    return data


def capture_jpeg_data_uri(media, max_width: int = 768, quality: int = 70,
                          blur: bool = True, method: str = DEFAULT_METHOD,
                          runtime: str = DEFAULT_RUNTIME,
                          style: str = DEFAULT_STYLE) -> str | None:
    """Grab one camera frame and return a `data:image/jpeg;base64,...` URI.

    With `blur` (the default), detected faces are obscured before the
    frame is encoded, so no unblurred image ever leaves the robot.
    Returns None if no frame or if OpenCV is unavailable.
    """
    data = capture_jpeg(media, max_width=max_width, quality=quality, blur=blur,
                        method=method, runtime=runtime, style=style)
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

    def __init__(self, media, rate_hz: float = 8.0,
                 method: str = DEFAULT_METHOD, runtime: str = DEFAULT_RUNTIME):
        super().__init__(name="FaceTracker", daemon=True)
        self.media = media
        self.period = 1.0 / rate_hz
        self.method, self.runtime = method, runtime
        self._stop = threading.Event()
        self.enabled = False

        self._lock = threading.Lock()
        self._offset = (0.0, 0.0)
        self._has_face = False
        self._last_seen = 0.0

    @property
    def available(self) -> bool:
        """True if the detector loaded, and so if tracking can run at all."""
        return available(self.method, self.runtime)

    def set_enabled(self, value: bool) -> None:
        if bool(value) != self.enabled:
            logger.info("face tracking %s", "enabled" if value else "disabled")
        self.enabled = bool(value)
        if not value:
            with self._lock:
                self._offset = (0.0, 0.0)
                self._has_face = False

    def get_offset(self) -> tuple[float, float, bool]:
        """Return the last offset and whether it is recent enough to act on."""
        with self._lock:
            fresh = self._has_face and (time.monotonic() - self._last_seen) < 1.0
            return self._offset[0], self._offset[1], fresh

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Poll the camera at `rate_hz` while enabled."""
        while not self._stop.is_set():
            # The detector can be swapped from the settings page, so being
            # unavailable now is a reason to idle, not to exit.
            if not self.enabled or not self.available:
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
        """Find the largest face in one frame and store its offset."""
        if frame.ndim != 3:
            return
        h, w = frame.shape[:2]
        faces = detect_faces(frame, self.method, self.runtime)
        if not faces:
            with self._lock:
                self._has_face = False
            return
        # Largest face wins.
        x, y, fw, fh = max(faces, key=lambda f: f.box[2] * f.box[3]).box
        # Normalize: face right of center -> positive offset_x.
        off_x = (x + fw / 2.0 - w / 2.0) / (w / 2.0)
        off_y = (y + fh / 2.0 - h / 2.0) / (h / 2.0)
        with self._lock:
            self._offset = (float(np.clip(off_x, -1, 1)), float(np.clip(off_y, -1, 1)))
            self._has_face = True
            self._last_seen = time.monotonic()