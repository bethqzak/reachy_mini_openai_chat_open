"""Face detection: several detectors, several runtimes, one result type.

    faces = detect_faces(frame)                     # yunet on opencv
    faces = detect_faces(frame, "haar")             # a different detector
    faces = detect_faces(frame, runtime="coreml")   # a different runtime

`method` chooses what finds faces, `runtime` chooses what executes the model.
Register a new one in DETECTORS or RUNTIMES and it becomes selectable; nothing
else in the pipeline has to change.
"""

from __future__ import annotations

import functools
import platform
import queue
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .log import get_logger

logger = get_logger("detection")

try:
    import cv2  # type: ignore

    _HAVE_CV2 = True
except Exception as _cv2_err:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAVE_CV2 = False
    logger.warning("OpenCV (cv2) not available: %s — detection disabled", _cv2_err)

Box = tuple[float, float, float, float]  # x, y, w, h


class Face(NamedTuple):
    """One detected face. `landmarks` is 5 points as (5, 2), or None."""

    box: Box
    score: float
    landmarks: np.ndarray | None = None
    # The box the blur is sized from, when it differs from the one above.
    shape_box: Box | None = None


# ====================================================================== #
# YuNet: a mirror-padded frame scanned by several passes, whose detections
# are then pooled into one cluster per face and put to a vote.
# Tuned to over-blur: a missed face leaks, a false positive costs a patch.
# ====================================================================== #

_STATIC = Path(__file__).with_name("static")
_YUNET_MODEL = _STATIC / "face_detection_yunet_2023mar.onnx"
# The same weights re-exported with a free input size, so onnxruntime can run
# a frame at its own resolution. A provider that rejects dynamic shapes uses
# the original export instead, which is fixed at 640 and must be letterboxed.
_YUNET_DYNAMIC = _STATIC / "face_detection_yunet_2023mar_dynamic.onnx"
_YUNET_FIXED_PX = 640
_YUNET_NMS_THRESHOLD = 0.3
_YUNET_SCORE_FLOOR = 0.3        # detections below this are ignored
_YUNET_SCORE_CONFIDENT = 0.85   # detections above this are kept unconditionally
_PAD_FRACTION = 0.15            # mirrored border added before detection
_ROTATIONS = (-30.0, 30.0)      # extra passes on a rotated frame, for tilt
_UPSCALE_BELOW_PX = 1200        # only frames narrower than this get the extra pass
_UPSCALE_FACTOR = 1.5           # how far that pass enlarges the frame

# grouping
_CLUSTER_IOU = 0.3              # overlap at which two boxes are the same face
_CONTAINMENT = 0.6              # or how far one box sits inside another
_CENTRE_TOLERANCE_FRACTION = 0.03  # how far out of frame a face may be centred
# Shared with the blurring, which has to know the same faces are clipped.
EDGE_MARGIN_FRACTION = 0.02     # how near an edge counts as a clipped face
_EDGE_MIN_SCORE = 0.75          # score a clipped face needs to be kept
_WEAK_SCORE = 0.60              # below this a rotated pass cannot vouch alone

_POOL_SIZE = 4                  # detectors, and so passes that may run at once

# Haar
_HAAR_CASCADES = ("haarcascade_frontalface_default.xml",
                  "haarcascade_profileface.xml")
_HAAR_MIN_SIZE = (24, 24)
_HAAR_SCALE_FACTOR = 1.1
_HAAR_MIN_NEIGHBOURS = 4


# -- runtimes: what actually executes the YuNet model ------------------ #
# Each returns raw detections as an (N, 15) array: x, y, w, h, 10 landmark
# coordinates, score. Everything above this layer is shared.

class _Runtime:
    """A way of running YuNet. Subclasses fill in `_run` and `available`."""

    name = "?"
    platforms = ("any",)
    note = ""

    def available(self) -> bool:
        return False

    def run(self, bgr: np.ndarray) -> np.ndarray:
        try:
            return self._run(bgr)
        except Exception:
            logger.exception("%s detection failed", self.name)
            return np.empty((0, 15), np.float32)

    def _run(self, bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _OpenCVRuntime(_Runtime):
    """OpenCV's own FaceDetectorYN. Runs at any input size."""

    name = "opencv"
    platforms = ("windows", "linux", "darwin")
    note = "any input size, no extra dependency"

    def available(self) -> bool:
        return bool(_opencv_detectors())

    def _run(self, bgr: np.ndarray) -> np.ndarray:
        free = _idle_detectors()
        det = free.get()
        try:
            det.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = det.detect(bgr)
        finally:
            free.put(det)
        return np.empty((0, 15), np.float32) if faces is None else np.asarray(faces)


class _OrtRuntime(_Runtime):
    """YuNet through onnxruntime on a named execution provider.

    Prefers the dynamic re-export, which runs at the frame's own size. A
    provider that rejects dynamic shapes falls back to the fixed square, into
    which frames are letterboxed; that costs small-face sensitivity.
    """

    provider = "CPUExecutionProvider"

    def available(self) -> bool:
        try:
            import onnxruntime as ort
        except Exception:
            return False
        return (_YUNET_DYNAMIC.exists() and _YUNET_MODEL.exists()
                and self.provider in ort.get_available_providers())

    def _session(self, model: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 4   # the dynamic-shape probe may log a failure
        providers = list(dict.fromkeys([self.provider, "CPUExecutionProvider"]))
        return ort.InferenceSession(str(model), opts, providers=providers)

    @functools.cached_property
    def _engine(self):
        """(session, letterbox) — whether this provider took the dynamic model."""
        try:
            sess = self._session(_YUNET_DYNAMIC)
            probe = np.zeros((1, 3, 64, 64), np.float32)
            sess.run(None, {sess.get_inputs()[0].name: probe})
            return sess, False
        except Exception:
            logger.info("%s rejects dynamic shapes — letterboxing into %dpx",
                        self.name, _YUNET_FIXED_PX)
            return self._session(_YUNET_MODEL), True

    def _run(self, bgr: np.ndarray) -> np.ndarray:
        sess, letterbox = self._engine
        h, w = bgr.shape[:2]
        if letterbox:
            scale = min(_YUNET_FIXED_PX / w, _YUNET_FIXED_PX / h)
            ih = iw = _YUNET_FIXED_PX
        else:
            # The network halves the frame five times, so both sides must be a
            # multiple of 32; the surplus is left black.
            scale = 1.0
            ih, iw = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        canvas = np.zeros((ih, iw, 3), np.uint8)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas[:nh, :nw] = (bgr if scale == 1.0 else
                            cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR))
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: blob})
        faces = _decode_yunet(out, ih, iw)
        if scale != 1.0:
            faces[:, :14] /= scale
        return faces


class _CoreMLRuntime(_OrtRuntime):
    name = "coreml"
    platforms = ("darwin",)
    provider = "CoreMLExecutionProvider"
    note = "Apple Neural Engine; fast, but 640px letterbox"


class _DirectMLRuntime(_OrtRuntime):
    name = "directml"
    platforms = ("windows",)
    provider = "DmlExecutionProvider"
    note = "any DirectX 12 GPU; the Windows choice"


class _OpenVINORuntime(_OrtRuntime):
    name = "openvino"
    platforms = ("windows", "linux")
    provider = "OpenVINOExecutionProvider"
    note = "Intel CPU/iGPU/NPU; the x86 choice without a discrete GPU"


class _OrtCPURuntime(_OrtRuntime):
    name = "onnx-cpu"
    platforms = ("windows", "linux", "darwin")
    provider = "CPUExecutionProvider"
    note = "portable, runs at full frame resolution"


RUNTIMES: dict[str, _Runtime] = {r.name: r for r in (
    _OpenCVRuntime(), _CoreMLRuntime(), _DirectMLRuntime(),
    _OpenVINORuntime(), _OrtCPURuntime())}


def _decode_yunet(out: list, h: int, w: int) -> np.ndarray:
    """Turn YuNet's raw tensors into (N, 15) rows, the shape OpenCV returns.

    Outputs are cls, obj, bbox and kps at strides 8, 16 and 32; a box is the
    offset of its cell centre plus a log-scaled size.
    """
    rows = []
    for i, stride in enumerate((8, 16, 32)):
        cls = np.asarray(out[i]).reshape(-1)
        obj = np.asarray(out[i + 3]).reshape(-1)
        bbox = np.asarray(out[i + 6]).reshape(-1, 4)
        kps = np.asarray(out[i + 9]).reshape(-1, 10)
        score = np.sqrt(np.clip(cls * obj, 0.0, 1.0))
        keep = score >= _YUNET_SCORE_FLOOR
        if not keep.any():
            continue
        cols, grid_h = w // stride, h // stride
        cx = np.tile(np.arange(cols, dtype=np.float32), grid_h)[keep]
        cy = np.repeat(np.arange(grid_h, dtype=np.float32), cols)[keep]
        b, k = bbox[keep], kps[keep]
        bw, bh = np.exp(b[:, 2]) * stride, np.exp(b[:, 3]) * stride
        mx, my = (cx + b[:, 0]) * stride, (cy + b[:, 1]) * stride
        points = np.empty((len(k), 10), np.float32)
        points[:, 0::2] = (cx[:, None] + k[:, 0::2]) * stride
        points[:, 1::2] = (cy[:, None] + k[:, 1::2]) * stride
        rows.append(np.column_stack([mx - bw / 2, my - bh / 2, bw, bh,
                                     points, score[keep]]))
    if not rows:
        return np.empty((0, 15), np.float32)
    rows = np.vstack(rows).astype(np.float32)
    keep = cv2.dnn.NMSBoxes(rows[:, :4].tolist(), rows[:, 14].tolist(),
                            _YUNET_SCORE_FLOOR, _YUNET_NMS_THRESHOLD)
    return rows[np.asarray(keep).reshape(-1)] if len(keep) else np.empty((0, 15), np.float32)


# -- the OpenCV detector pool ------------------------------------------ #

@functools.cache
def _opencv_detectors() -> tuple:
    """Build every FaceDetectorYN instance, or () if none can be built.

    One per pass: a detector holds its input size as state, so two passes can
    only run at the same time if each has its own.
    """
    if not _HAVE_CV2 or not hasattr(cv2, "FaceDetectorYN"):
        logger.warning("OpenCV build has no FaceDetectorYN — YuNet unavailable")
    elif not _YUNET_MODEL.exists():
        logger.warning("YuNet model missing at %s", _YUNET_MODEL)
    else:
        try:
            return tuple(cv2.FaceDetectorYN.create(
                str(_YUNET_MODEL), "", (320, 320),
                _YUNET_SCORE_FLOOR, _YUNET_NMS_THRESHOLD, 5000)
                for _ in range(_POOL_SIZE))
        except Exception:
            logger.exception("could not load YuNet model %s", _YUNET_MODEL)
    return ()


@functools.cache
def _idle_detectors() -> queue.Queue:
    """The detectors not currently in use by a thread."""
    free: queue.Queue = queue.Queue()
    for d in _opencv_detectors():
        free.put(d)
    return free


@functools.cache
def _pass_pool() -> ThreadPoolExecutor:
    """Threads the detection passes run on."""
    return ThreadPoolExecutor(_POOL_SIZE, thread_name_prefix="detect")


# -- detection passes --------------------------------------------------- #

class _Candidate(NamedTuple):
    """One detection, tagged with the pass that produced it."""

    box: Box
    score: float
    pass_id: str
    landmarks: np.ndarray | None = None


def _run_pass(runtime: _Runtime, bgr: np.ndarray, pass_id: str) -> list[_Candidate]:
    """Run one image through the model and label what it finds."""
    return [_Candidate((float(f[0]), float(f[1]), float(f[2]), float(f[3])),
                       float(f[14]), pass_id,
                       np.asarray(f[4:14], np.float64).reshape(5, 2))
            for f in runtime.run(bgr)]


def _rescale(cands: list[_Candidate], factor: float) -> list[_Candidate]:
    """Map candidates found on a resized copy back to the original scale."""
    return [_Candidate((x / factor, y / factor, bw / factor, bh / factor),
                       score, pid, None if m is None else m / factor)
            for (x, y, bw, bh), score, pid, m in cands]


def _pass_rotated(runtime, bgr: np.ndarray, angle: float) -> list[_Candidate]:
    """Detect on a rotated copy and map the results back to the upright frame."""
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    # The canvas keeps its size rather than growing to fit the corners: the
    # mirrored border already covers them.
    rot = cv2.warpAffine(bgr, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    inv = cv2.invertAffineTransform(m)
    out = []
    for (x, y, bw, bh), score, pid, marks in _run_pass(runtime, rot, f"rot{angle:+.0f}"):
        # Only the centre is mapped back; the box keeps its detected size.
        ox, oy = inv @ np.array([x + bw / 2.0, y + bh / 2.0, 1.0])
        if marks is not None:
            marks = np.hstack([marks, np.ones((len(marks), 1))]) @ inv.T
        out.append(_Candidate((float(ox - bw / 2.0), float(oy - bh / 2.0), bw, bh),
                              score, pid, marks))
    return out


def _pass_upscaled(runtime, padded: np.ndarray) -> list[_Candidate]:
    """Detect on an enlarged copy, for faces too small to find at native scale."""
    h, w = padded.shape[:2]
    big = cv2.resize(padded, (int(w * _UPSCALE_FACTOR), int(h * _UPSCALE_FACTOR)),
                     interpolation=cv2.INTER_LINEAR)
    return _rescale(_run_pass(runtime, big, "upscaled"), _UPSCALE_FACTOR)


def _detect_passes(runtime, padded: np.ndarray) -> list[_Candidate]:
    """Run every pass over the frame at once and pool the results."""
    # The passes are independent, so they run on separate threads; both
    # OpenCV and onnxruntime release the GIL for the duration.
    jobs = [partial(_run_pass, runtime, padded, "plain")]
    if padded.shape[1] < _UPSCALE_BELOW_PX:
        jobs.append(partial(_pass_upscaled, runtime, padded))
    jobs += [partial(_pass_rotated, runtime, padded, a) for a in _ROTATIONS]
    return [c for found in _pass_pool().map(lambda job: job(), jobs) for c in found]


# -- grouping: pooling the passes into one cluster per face ------------- #

def _on_frame(x: float, y: float, w: int, h: int) -> bool:
    """True if a point sits on the picture rather than in the mirrored border."""
    tol = _CENTRE_TOLERANCE_FRACTION * max(w, h)
    return -tol <= x < w + tol and -tol <= y < h + tol


def _overlaps(a: Box, b: Box) -> bool:
    """True if two boxes overlap enough to be the same face."""
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if ix <= 0.0 or iy <= 0.0:
        return False
    inter = ix * iy
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    # A small box well inside a larger one counts, so that two people standing
    # shoulder to shoulder are not merged on overlap alone.
    smaller = min(area_a, area_b)
    if smaller > 0 and inter / smaller >= _CONTAINMENT:
        return True
    union = area_a + area_b - inter
    return union > 0 and inter / union >= _CLUSTER_IOU


class _Cluster(NamedTuple):
    """Detections of one face, pooled across passes."""

    box: Box                        # union of the contributing boxes
    score: float                    # best score seen
    passes: frozenset               # which passes contributed
    landmarks: np.ndarray | None
    shape_box: Box | None = None    # representative's box; the union is inflated


class _Group:
    """A cluster while it is still being pooled."""

    __slots__ = ("x0", "y0", "x1", "y1", "score", "passes", "marks", "rep")

    def __init__(self, cand: _Candidate, on_frame: bool):
        x, y, w, h = cand.box
        self.x0, self.y0, self.x1, self.y1 = x, y, x + w, y + h
        self.score = cand.score
        self.passes = {cand.pass_id}
        self.marks = cand.landmarks if on_frame else None
        self.rep = cand.box if on_frame else None

    @property
    def box(self) -> Box:
        return (self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)

    def absorb(self, other: "_Group") -> None:
        """Pool another group in, taking its landmarks if this one has none."""
        self.x0, self.y0 = min(self.x0, other.x0), min(self.y0, other.y0)
        self.x1, self.y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        self.passes |= other.passes
        if self.marks is None and other.marks is not None:
            self.score = other.score
            self.marks, self.rep = other.marks, other.rep
        elif other.score > self.score and (other.marks is not None
                                           or self.marks is None):
            self.score = other.score

    def add(self, cand: _Candidate, on_frame: bool) -> None:
        """Pool a single detection in."""
        self.absorb(_Group(cand, on_frame))

    def freeze(self) -> _Cluster:
        return _Cluster(self.box, self.score, frozenset(self.passes),
                        self.marks, self.rep)


def _cluster(cands: list[_Candidate], w: int, h: int) -> list[_Cluster]:
    """Group detections of the same face together."""
    # Detections whose landmarks land on the picture sort first, so they seed
    # the groups and own the size: a reflected twin can outscore a real face.
    # Among those the plain pass goes first, since it is the only one that sees
    # the frame undistorted.
    ordered = [(c, c.landmarks is not None and len(c.landmarks) > 0
                and _on_frame(*c.landmarks.mean(axis=0), w, h)) for c in cands]
    ordered.sort(key=lambda p: (not p[1], p[0].pass_id != "plain", -p[0].score))

    groups: list[_Group] = []
    for cand, on_frame in ordered:
        for g in groups:
            if _overlaps(cand.box, g.box):
                g.add(cand, on_frame)
                break
        else:
            groups.append(_Group(cand, on_frame))

    # Repeated until nothing merges: one sweep leaves chains, where A merged
    # into B and C overlaps A but was only ever tested against B.
    i = 0
    while i < len(groups):
        j = i + 1
        while j < len(groups):
            if _overlaps(groups[i].box, groups[j].box):
                groups[i].absorb(groups.pop(j))
                j = i + 1
            else:
                j += 1
        i += 1
    return [g.freeze() for g in groups]


def _plausible(cands: list[_Candidate], w: int, h: int) -> list[_Cluster]:
    """Cluster the detections and keep only those corroborated by two passes."""
    margin = EDGE_MARGIN_FRACTION * max(w, h)
    keep = []
    for c in _cluster(cands, w, h):
        x, y, bw, bh = c.box
        at_edge = (x <= margin or y <= margin
                   or x + bw >= w - margin or y + bh >= h - margin)
        # A real face is normally found by the plain pass. A face at an edge is
        # excused that, but then has to be a confident detection: at a lower
        # bar the warped passes will vouch for a chair or a blank wall.
        corroborated = len(c.passes) >= 2 and (
            "plain" in c.passes or (at_edge and c.score >= _EDGE_MIN_SCORE))
        # A weak detection needs an unwarped second opinion. Rotating the frame
        # distorts it enough to make an arm or a hand read as a face.
        warped_vouch = c.score < _WEAK_SCORE and "upscaled" not in c.passes
        if (corroborated and not warped_vouch) or c.score >= _YUNET_SCORE_CONFIDENT:
            keep.append(c)
    return keep


def _mirror(lo: float, size: float, limit: int) -> float | None:
    """Reflect a span lying wholly outside 0..limit back across the border,
    or None if any of it is inside."""
    if lo >= limit:
        return 2.0 * limit - (lo + size)
    if lo + size <= 0:
        return -(lo + size)
    return None


def _fold_reflection(c: _Cluster, w: int, h: int) -> Face:
    """Move a blur that landed off the picture onto the face it mirrors."""
    # A face in profile often scores best as its own reflection in the border.
    # Landmarks are dropped: mirroring swaps left and right.
    x, y, bw, bh = c.shape_box or c.box
    nx, ny = _mirror(x, bw, w), _mirror(y, bh, h)
    if nx is None and ny is None:
        return Face(c.box, c.score, c.landmarks, c.shape_box)
    box = (x if nx is None else nx, y if ny is None else ny, bw, bh)
    return Face(box, c.score, None, box)


# -- the detectors ------------------------------------------------------ #

def _detect_yunet(bgr: np.ndarray, runtime: _Runtime) -> list[Face]:
    """Several passes over a mirror-padded frame, pooled and put to a vote.

    Not a guarantee: a head seen from behind has no face to find.
    """
    h, w = bgr.shape[:2]
    # Mirrored border, so a half-visible face at an edge is completed.
    pad = int(_PAD_FRACTION * max(h, w))
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

    # Back to frame coordinates, dropping anything centred in the border.
    on_frame = []
    for (x, y, bw, bh), score, pid, marks in _detect_passes(runtime, padded):
        x, y = x - pad, y - pad
        if _on_frame(x + bw / 2.0, y + bh / 2.0, w, h):
            on_frame.append(_Candidate((x, y, bw, bh), score, pid,
                                       None if marks is None else marks - pad))
    return [_fold_reflection(c, w, h) for c in _plausible(on_frame, w, h)]


@functools.cache
def _haar_cascades() -> tuple:
    """Frontal and profile cascades, loaded once."""
    out = []
    if _HAVE_CV2:
        for name in _HAAR_CASCADES:
            try:
                c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
                if not c.empty():
                    out.append((name, c))
                else:
                    logger.warning("cascade empty/missing: %s", name)
            except Exception:
                logger.exception("could not load cascade %s", name)
    return tuple(out)


def _detect_haar(bgr: np.ndarray, runtime: _Runtime) -> list[Face]:
    """Haar cascades: frontal, plus profile run both ways round.

    The profile cascade only detects one facing direction, so it is run on the
    mirrored frame too. No landmarks and no score, so the blur is sized from
    the box alone.
    """
    cascades = _haar_cascades()
    if not cascades:
        return []
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = []
    for name, cascade in cascades:
        views = [(gray, False)]
        if "profile" in name:
            views.append((cv2.flip(gray, 1), True))
        for view, mirrored in views:
            for (x, y, fw, fh) in cascade.detectMultiScale(
                    view, _HAAR_SCALE_FACTOR, _HAAR_MIN_NEIGHBOURS,
                    minSize=_HAAR_MIN_SIZE):
                if mirrored:
                    x = w - x - fw
                box = (float(x), float(y), float(fw), float(fh))
                faces.append(Face(box, 1.0, None, box))
    return faces


# name -> (what detects, whether it can run on a given runtime). Haar carries
# its own cascades, so the runtime it is handed makes no difference to it.
DETECTORS = {"yunet": (_detect_yunet, lambda r: bool(r and r.available())),
             "haar": (_detect_haar, lambda _: bool(_haar_cascades()))}
DEFAULT_METHOD, DEFAULT_RUNTIME = "yunet", "onnx-cpu"


def available(method: str = DEFAULT_METHOD, runtime: str = DEFAULT_RUNTIME) -> bool:
    """True if this combination can run here."""
    spec = DETECTORS.get(method)
    return bool(_HAVE_CV2 and spec and spec[1](RUNTIMES.get(runtime)))


def detect_faces(bgr: np.ndarray, method: str = DEFAULT_METHOD,
                 runtime: str = DEFAULT_RUNTIME) -> list[Face] | None:
    """Find every face in a BGR frame.

    Returns the faces found, or None if the requested detector cannot run at
    all — which callers must tell apart from a frame with nobody in it.
    """
    if not _HAVE_CV2 or bgr is None or bgr.ndim != 3:
        return None
    spec = DETECTORS.get(method)
    if spec is None:
        logger.warning("unknown detection method %r", method)
        return None
    fn, ready = spec
    r = RUNTIMES.get(runtime)
    if not ready(r):
        logger.warning("detection method %r unavailable on runtime %r",
                       method, runtime)
        return None
    return fn(bgr, r)


def this_platform() -> str:
    """'darwin', 'windows' or 'linux'."""
    return {"Darwin": "darwin", "Windows": "windows"}.get(platform.system(), "linux")


def runtime_options() -> list[dict]:
    """Every runtime with whether it can run here, for a UI to offer."""
    here = this_platform()
    return [{"name": r.name, "platforms": r.platforms, "note": r.note,
             "supported": here in r.platforms or "any" in r.platforms,
             "available": r.available()} for r in RUNTIMES.values()]
