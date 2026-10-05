"""Face blurring: an oval fitted to each face, filled by one of several styles.

    blur_faces(frame, faces)              # gaussian
    blur_faces(frame, faces, "mosaic")

The oval fitting, the masking and the blending are shared; a style only says
how one rectangular patch is obscured. Register a new one in STYLES.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .detection import EDGE_MARGIN_FRACTION, Box, Face
from .log import get_logger

logger = get_logger("blur")

try:
    import cv2  # type: ignore

    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAVE_CV2 = False

# oval size
_BOX_COVER = 1.16               # blur size sideways and up, vs half the box
_COVER_DOWN = 0.92              # the same below the chin
_FEATURE_MARGIN = 0.55          # how far past each landmark the blur must reach
_MAX_GROWTH = 1.35              # most one landmark may grow the blur
_MAX_WIDTH_RATIO = 1.00         # widest the blur may be against its height
_WIDTH_TRADE = 0.5              # share of width lost to that, added back on top
_NEIGHBOUR_SHARE = 0.62         # share of the gap to the next face a blur fills
_NEIGHBOUR_FLOOR = 0.68         # smallest a neighbour may shrink a blur

# rendering
_GAUSSIAN_DIVISOR = 4.0         # blur strength: face size over this
_BLUR_WORK_SIGMA = 3.0          # sigma the gaussian is actually computed at
_MOSAIC_BLOCKS = 8              # mosaic cells across the shorter side of a face
_SOLID_CORE = 0.90              # share of the radius obscured at full strength
_OUTLINE_WAVE = 0.05            # how far the outline waves off a true ellipse

# CIE-Lab distance from a face's own centre colour, used to stop the blur at an
# object crossing the face. Not skin detection.
_DIFF_CUT_LOW = 18.0            # below this nothing is held back
_DIFF_CUT_HIGH = 34.0           # above this the blur is withheld entirely
_DIFF_LIGHT_WEIGHT = 0.15       # how much lightness counts towards the distance
_DIFF_CORE_EXTRA = 25.0         # extra tolerance at the middle of the face
_PROTECT_CORE = 0.72            # how far out that tolerance reaches
_PROTECT_FADE = 0.42            # the distance it fades over
_DIFF_REFERENCE = 0.40          # radius the reference colour is averaged over


# -- shaping: fitting an oval to each face ------------------------------ #

class Shape(NamedTuple):
    """One face's blur: an oval with four radii, measured in the face's own
    rotated frame so it can be warped rather than merely stretched."""

    centre: tuple[int, int]
    left: float
    right: float
    up: float
    down: float
    angle: float

    @property
    def reach(self) -> float:
        return max(self.left, self.right, self.up, self.down)


# Indices into the working list of radii, in Shape's own order.
_L, _R, _U, _D = 0, 1, 2, 3


def _face_frame(angle: float) -> tuple[float, float]:
    """Return the cosine and sine that rotate a frame offset into a face."""
    rad = np.radians(-angle)
    return float(np.cos(rad)), float(np.sin(rad))


def _clip_to_frame(box: Box, frame: tuple[int, int]) -> Box:
    """Return the part of a detection box that lies on the picture."""
    fw, fh = frame
    bx, by, bbw, bbh = box
    x, y = max(0.0, bx), max(0.0, by)
    bw = min(float(fw), bx + bbw) - x
    bh = min(float(fh), by + bbh) - y
    return box if bw <= 1.0 or bh <= 1.0 else (x, y, bw, bh)


def _box_radii(box: Box, clipped: Box) -> tuple[float, float, list[float]]:
    """Return the oval's centre and starting radii, from the box alone."""
    bx, by, bbw, bbh = box
    x, y, bw, bh = clipped
    cx, cy = x + bw / 2.0, y + bh / 2.0
    half_x, half_y = bw * 0.5, bh * 0.5
    # A side running off the picture reaches as far as the uncut box did.
    return cx, cy, [_BOX_COVER * max(half_x, cx - bx),
                    _BOX_COVER * max(half_x, bx + bbw - cx),
                    _BOX_COVER * max(half_y, cy - by),
                    _COVER_DOWN * max(half_y, by + bbh - cy)]


def _eye_line(marks: np.ndarray) -> tuple[float, float]:
    """Return the face's tilt in degrees and the span between the eyes."""
    right_eye, left_eye = marks[0], marks[1]
    dx, dy = left_eye - right_eye
    span = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx))) if span > 1.0 else 0.0
    return angle, span or 4.0


def _grow_to_landmarks(rads: list[float], marks: np.ndarray,
                       centre: tuple[float, float], angle: float,
                       eye_span: float, frame: tuple[int, int]) -> None:
    """Grow the radii in place until every facial feature is covered."""
    fw, fh = frame
    cx, cy = centre
    cos_a, sin_a = _face_frame(angle)
    centroid = marks[:5].mean(axis=0)
    caps = [r * _MAX_GROWTH for r in rads]
    for _ in range(4):      # repeated passes settle points that need both axes
        for point in marks[:5]:
            # Push each point outward first: an eye is a feature, not a dot.
            away = point - centroid
            norm = float(np.hypot(*away))
            edge = (point + away / norm * (eye_span * _FEATURE_MARGIN)
                    if norm > 1e-3 else point)
            # A feature off the picture needs no covering.
            if not (0.0 <= float(edge[0]) < fw and 0.0 <= float(edge[1]) < fh):
                continue
            ox, oy = float(edge[0]) - cx, float(edge[1]) - cy
            u = ox * cos_a - oy * sin_a
            v = ox * sin_a + oy * cos_a
            i, j = (_R if u >= 0 else _L), (_D if v >= 0 else _U)
            # How far outside the solid core the feature sits, if at all.
            across = abs(u) / max(rads[i], 1e-3)
            along = abs(v) / max(rads[j], 1e-3)
            need = float(np.hypot(across, along)) / _SOLID_CORE
            if need <= 1.0:
                continue
            # Split the growth between the axes by which one it overruns.
            total = across * across + along * along
            share = (across * across) / total if total > 1e-9 else 0.5
            rads[i] = min(rads[i] * need ** share, caps[i])
            rads[j] = min(rads[j] * need ** (1.0 - share), caps[j])


def _limit_width(rads: list[float], cx: float, fw: int) -> None:
    """Narrow an oval wider than a head, in place, keeping it centred."""
    span_y = rads[_U] + rads[_D]
    # Measured on the width actually in frame, so a side-clipped face is not
    # narrowed for a width that is never drawn.
    span_x = min(rads[_L], max(0.0, cx)) + min(rads[_R], max(0.0, fw - cx))
    if span_x > span_y * _MAX_WIDTH_RATIO and span_x > 1e-3:
        squeeze = (span_y * _MAX_WIDTH_RATIO) / span_x
        rads[_L] *= squeeze
        rads[_R] *= squeeze
        # Give some of the lost width back upward, onto forehead and hair.
        rads[_U] += span_x * (1.0 - squeeze) * _WIDTH_TRADE


def _extend_past_edges(rads: list[float], clipped: Box,
                       centre: tuple[float, float], frame: tuple[int, int]) -> None:
    """Push the radii past any edge cutting the face, in place."""
    # Far enough that the border sits inside the solid core, or the taper fades
    # out early and leaves a strip of cheek along it.
    fw, fh = frame
    cx, cy = centre
    x, y, bw, bh = clipped
    margin = EDGE_MARGIN_FRACTION * max(fw, fh)
    if x <= margin:
        rads[_L] = max(rads[_L], (cx + 1.0) / _SOLID_CORE)
    if x + bw >= fw - margin:
        rads[_R] = max(rads[_R], (fw - cx + 1.0) / _SOLID_CORE)
    if y <= margin:
        rads[_U] = max(rads[_U], (cy + 1.0) / _SOLID_CORE)
    if y + bh >= fh - margin:
        rads[_D] = max(rads[_D], (fh - cy + 1.0) / _SOLID_CORE)


def _recentre_axis(c: float, rads: list[float], lo: int, hi: int,
                   want: float, size: int) -> float:
    """Return `c` moved to `want` along one axis, the radii absorbing the move."""
    shift = c - want
    if abs(shift) > 1.0:
        shift = float(np.clip(shift, c - size + 4.0, c - 4.0))
        if rads[lo] - shift > 1.0 and rads[hi] + shift > 1.0:
            rads[lo], rads[hi] = rads[lo] - shift, rads[hi] + shift
            return c - shift
    return c


def _recentre(centre: tuple[float, float], rads: list[float], box: Box,
              frame: tuple[int, int]) -> tuple[float, float]:
    """Return the centre moved back to the middle of the uncut face."""
    # Clipping the box drags the centre towards whichever edge cut it.
    cx, cy = centre
    fw, fh = frame
    bx, by, bbw, bbh = box
    return (_recentre_axis(cx, rads, _L, _R, bx + bbw / 2.0, fw),
            _recentre_axis(cy, rads, _U, _D, by + bbh / 2.0, fh))


def face_shape(face: Face, frame: tuple[int, int]) -> Shape:
    """Fit a blur oval to one face."""
    fw = frame[0]
    box = face.shape_box or face.box
    clipped = _clip_to_frame(box, frame)
    cx, cy, rads = _box_radii(box, clipped)

    # Without landmarks the box alone has to do.
    marks = face.landmarks
    if marks is None or len(marks) < 5 or clipped[2] < 2.0 or clipped[3] < 2.0:
        return Shape((int(round(cx)), int(round(cy))), *rads, 0.0)

    angle, eye_span = _eye_line(marks)
    _grow_to_landmarks(rads, marks, (cx, cy), angle, eye_span, frame)
    _limit_width(rads, cx, fw)
    cx, cy = _recentre((cx, cy), rads, box, frame)
    _extend_past_edges(rads, clipped, (cx, cy), frame)
    return Shape((int(round(cx)), int(round(cy))), *rads, angle)


def _neighbour_caps(shapes: list[Shape]) -> list[float]:
    """Return a shrink factor per face so neighbouring blurs don't overlap."""
    def cap(a: Shape) -> float:
        if a.reach <= 1:
            return 1.0
        gaps = [d for b in shapes if b is not a
                and (d := float(np.hypot(a.centre[0] - b.centre[0],
                                         a.centre[1] - b.centre[1]))) > 1]
        return (min(gaps, default=np.inf) * _NEIGHBOUR_SHARE) / a.reach

    return [min(1.0, max(_NEIGHBOUR_FLOOR, cap(a))) for a in shapes]


# -- styles: a way of obscuring a patch, and a shape to apply it through -- #

def _gaussian_patch(roi: np.ndarray) -> np.ndarray:
    """Blur a patch past recognition."""
    h, w = roi.shape[:2]
    sigma = max(2.0, min(w, h) / _GAUSSIAN_DIVISOR)
    # A blur this heavy needs a kernel as wide as the face, which is far more
    # work than the result is worth, so shrink first and blur the small copy.
    # INTER_AREA is itself an average, so the shrink does part of the blurring.
    step = max(1, int(sigma / _BLUR_WORK_SIGMA))
    small = roi if step == 1 else cv2.resize(
        roi, (max(2, w // step), max(2, h // step)), interpolation=cv2.INTER_AREA)
    sigma /= step
    k = int(sigma * 4) | 1
    blurred = cv2.GaussianBlur(small, (k, k), sigma)
    return blurred if step == 1 else cv2.resize(blurred, (w, h),
                                                interpolation=cv2.INTER_LINEAR)


def _mosaic_patch(roi: np.ndarray) -> np.ndarray:
    """Reduce a patch to a grid of flat blocks."""
    h, w = roi.shape[:2]
    step = max(1, min(w, h) // _MOSAIC_BLOCKS)
    small = cv2.resize(roi, (max(1, w // step), max(1, h // step)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# name -> (how a patch is obscured, what shape it is applied through). The
# outline style obscures nothing; it marks what was found instead.
STYLES = {
    "gaussian-ellipse": (_gaussian_patch, "ellipse"),
    "gaussian-square": (_gaussian_patch, "square"),
    "mosaic-ellipse": (_mosaic_patch, "ellipse"),
    "mosaic-square": (_mosaic_patch, "square"),
    "outline": (None, "outline"),
}
DEFAULT_STYLE = "gaussian-ellipse"


# -- rendering ---------------------------------------------------------- #

def _ramp(x: np.ndarray, width: float) -> np.ndarray:
    """Ease from 0 below 0 to 1 above `width`, with zero slope at both ends."""
    t = np.clip(x / max(width, 1e-3), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hold_back(roi: np.ndarray, r: np.ndarray) -> np.ndarray | None:
    """Return a 0..1 mask holding the blur off objects crossing the face.

    None if the face gives no reliable reference colour.
    """
    # The reference is the face's own centre, and the bar rises towards the
    # middle, where a lip or highlight would otherwise read as an object.
    core = r <= _DIFF_REFERENCE
    if int(core.sum()) <= 24:
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = lab - lab[core].mean(axis=0)
    delta[:, :, 0] *= _DIFF_LIGHT_WEIGHT
    dist = np.linalg.norm(delta, axis=2)
    low = _DIFF_CUT_LOW + _DIFF_CORE_EXTRA * _ramp(_PROTECT_CORE - r, _PROTECT_FADE)
    keep = 1.0 - np.clip((dist - low) / (_DIFF_CUT_HIGH - _DIFF_CUT_LOW), 0.0, 1.0)
    # Smoothed so the cut follows the object rather than speckling.
    k = max(3, (min(roi.shape[:2]) // 12) | 1)
    return cv2.GaussianBlur(keep, (k, k), 0)


def _ellipse_alpha(roi, shape, box, radii):
    """Mask for one oval: solid over the middle, tapering to nothing at the rim."""
    (x0, y0, x1, y1), (left, right, up, down) = box, radii
    ox, oy = shape.centre
    # Distance from the centre as a fraction of the radius on that side, in the
    # face's own frame, so each side has its own reach.
    cos_a, sin_a = _face_frame(shape.angle)
    dx = np.arange(x0, x1, dtype=np.float32) - ox
    dy = (np.arange(y0, y1, dtype=np.float32) - oy)[:, None]
    fx = dx * cos_a - dy * sin_a
    fy = dx * sin_a + dy * cos_a
    u = fx / np.where(fx >= 0, right, left)
    v = fy / np.where(fy >= 0, down, up)
    r = np.sqrt(u * u + v * v)

    # Wobble the outline, so it does not read as an imposed ellipse. The phase
    # comes from the position, so a face keeps the same outline.
    theta, seed = np.arctan2(v, u), ox * 0.7 + oy * 1.3
    r = r * (1.0 - _OUTLINE_WAVE * (np.sin(2.0 * theta + seed) * 0.5
                                    + np.sin(3.0 * theta - seed * 1.7) * 0.3
                                    + np.sin(5.0 * theta + seed * 0.4) * 0.2))
    soft = _ramp(1.0 - r, 1.0 - _SOLID_CORE)
    hold = _hold_back(roi, r)
    return soft if hold is None else soft * hold


def _draw_outlines(bgr, pairs) -> int:
    """Mark each face with a hollow red box and its score, obscuring nothing."""
    for shape, face in pairs:
        ox, oy = shape.centre
        x0, y0 = int(ox - shape.left), int(oy - shape.up)
        x1, y1 = int(ox + shape.right), int(oy + shape.down)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label = f"{face.score:.2f}"
        scale = max(0.4, min(1.0, (x1 - x0) / 180.0))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ty = max(th + 4, y0 - 4)
        cv2.rectangle(bgr, (x0, ty - th - 4), (x0 + tw + 6, ty + 2), (0, 0, 255), -1)
        cv2.putText(bgr, label, (x0 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 2)
    return len(pairs)


def blur_faces(bgr: np.ndarray, faces: list[Face] | None,
               style: str = DEFAULT_STYLE) -> int:
    """Obscure every face in the BGR frame, in place. Returns the count.

    Masks are accumulated and blended once, or two overlapping faces would each
    obscure the other's already-obscured pixels.
    """
    spec = STYLES.get(style)
    if spec is None:
        logger.warning("unknown blur style %r", style)
        return 0
    if not _HAVE_CV2 or not faces:
        return 0
    patch, kind = spec
    h, w = bgr.shape[:2]
    pairs = [(s, f) for s, f in ((face_shape(f, (w, h)), f) for f in faces)
             if s.reach >= 1]
    if not pairs:
        return 0
    if kind == "outline":
        return _draw_outlines(bgr, pairs)

    shapes = [s for s, _ in pairs]
    alpha = np.zeros((h, w), np.float32)
    layer = bgr.copy()
    touched = []
    for shape, cap in zip(shapes, _neighbour_caps(shapes)):
        ox, oy = shape.centre
        radii = tuple(max(1.0, v * cap) for v in
                      (shape.left, shape.right, shape.up, shape.down))

        # Work area: the oval is asymmetric, so bound it by its longest reach.
        reach = int(np.ceil(max(radii))) + 2
        x0, y0 = max(0, ox - reach), max(0, oy - reach)
        x1, y1 = min(w, ox + reach), min(h, oy + reach)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = bgr[y0:y1, x0:x1]
        layer[y0:y1, x0:x1] = patch(roi)
        if kind == "ellipse":
            soft = _ellipse_alpha(roi, shape, (x0, y0, x1, y1), radii)
        else:
            # A square covers the oval's own reach, hard-edged.
            soft = np.zeros((y1 - y0, x1 - x0), np.float32)
            left, right, up, down = radii
            soft[max(0, int(oy - up) - y0):int(oy + down) - y0,
                 max(0, int(ox - left) - x0):int(ox + right) - x0] = 1.0
        np.maximum(alpha[y0:y1, x0:x1], soft, out=alpha[y0:y1, x0:x1])
        touched.append((x0, y0, x1, y1))

    if not touched:
        return 0
    bx0, by0 = min(t[0] for t in touched), min(t[1] for t in touched)
    bx1, by1 = max(t[2] for t in touched), max(t[3] for t in touched)
    box = (slice(by0, by1), slice(bx0, bx1))
    a = alpha[box][:, :, None]
    bgr[box] = (layer[box] * a + bgr[box] * (1.0 - a)).astype(np.uint8)
    return len(touched)
