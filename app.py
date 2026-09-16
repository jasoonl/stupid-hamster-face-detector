from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

WINDOW_NAME = "make a face"
BOX_COLOR = (0, 255, 0)          # green tracking box, BGR
INK = (27, 27, 27)               # doodle line color, BGR
NOSE_COLOR = (161, 161, 226)     # soft pink, BGR
MOUTH_FILL = (63, 63, 107)       # open-mouth interior, BGR
TONGUE_COLOR = (170, 140, 235)   # tongue, BGR
PANEL_BG = (234, 242, 244)       # doodle-panel background, BGR (matches #f4f2ea)
FONT = cv2.FONT_HERSHEY_SIMPLEX

PANEL_W, PANEL_H = 480, 480      # each side of the combined window
CALIBRATION_FRAMES = 15

Box = Tuple[int, int, int, int]  # x, y, w, h

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

@dataclass
class Observation:
    box: Box
    eye_open: float       # 0..1, relative to calibrated baseline
    brow_raise: float     # 0..1
    mouth_open: float     # 0..1
    smile: float          # 0..1


@dataclass
class Baseline:
    eye_score: float = 1.0
    brow_row_frac: float = 0.35
    mouth_h: float = 3.0


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def detect_face(gray: np.ndarray) -> Optional[Box]:
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def eye_openness_score(gray: np.ndarray, face_box: Box) -> float:
    x, y, w, h = face_box
    band = gray[y + int(0.24 * h): y + int(0.52 * h), x + int(0.10 * w): x + int(0.90 * w)]
    if band.size == 0:
        return 0.0
    eyes = EYE_CASCADE.detectMultiScale(band, scaleFactor=1.1, minNeighbors=6, minSize=(12, 12))
    contrast = float(np.std(band))
    presence = min(len(eyes), 2) / 2.0
    return 0.65 * presence + 0.35 * min(contrast / 35.0, 1.0)


def darkest_row_frac(gray: np.ndarray, box_in_gray: Box) -> float:
    x, y, w, h = box_in_gray
    region = gray[y:y + h, x:x + w]
    if region.size == 0:
        return 0.5
    row_means = region.mean(axis=1)
    row = int(np.argmin(row_means))
    return row / max(1, h - 1)


def _contiguous_dip_width(profile: np.ndarray) -> int:
    """Width of the contiguous dark dip around a 1D profile's minimum,
    using a half-depth (FWHM-style) threshold. Robust to a few stray dark
    pixels in a way a raw "bounding box of dark pixels" is not — see
    test_facelogic.py for the real-photo test that caught this."""
    n = len(profile)
    if n == 0:
        return 0
    lo, mean = float(profile.min()), float(profile.mean())
    threshold = lo + (mean - lo) * 0.5
    center = int(np.argmin(profile))
    left = center
    while left > 0 and profile[left - 1] <= threshold:
        left -= 1
    right = center
    while right < n - 1 and profile[right + 1] <= threshold:
        right += 1
    return right - left + 1


def mouth_open_height(gray: np.ndarray, box_in_gray: Box) -> float:
    x, y, w, h = box_in_gray
    region = gray[y:y + h, x:x + w]
    if region.size == 0:
        return 1.0
    return float(_contiguous_dip_width(region.mean(axis=1)))


def smile_score(gray: np.ndarray, face_box: Box) -> float:
    """OpenCV's own trained smile cascade, run on the lower face — not a
    hand-rolled pixel heuristic. An earlier column-brightness-profile
    approach for "smile width" tested non-monotonic on a real photo (moved
    backwards as the simulated smile widened); a trained cascade sidesteps
    that class of problem entirely."""
    x, y, w, h = face_box
    band = gray[y + int(0.55 * h): y + h, x:x + w]
    if band.size == 0:
        return 0.0
    smiles = SMILE_CASCADE.detectMultiScale(
        band, scaleFactor=1.6, minNeighbors=25, minSize=(int(0.25 * w), int(0.10 * h))
    )
    if len(smiles) == 0:
        return 0.0
    widest = max(smiles, key=lambda s: s[2])
    return float(min(1.0, widest[2] / (0.7 * w)))


def measure(gray: np.ndarray, face_box: Box) -> dict:
    x, y, w, h = face_box
    eye_score = eye_openness_score(gray, face_box)

    brow_l = (x + int(0.16 * w), y + int(0.16 * h), int(0.30 * w), int(0.14 * h))
    brow_r = (x + int(0.54 * w), y + int(0.16 * h), int(0.30 * w), int(0.14 * h))
    brow_row_frac = (darkest_row_frac(gray, brow_l) + darkest_row_frac(gray, brow_r)) / 2.0

    mouth_box = (x + int(0.28 * w), y + int(0.68 * h), int(0.44 * w), int(0.22 * h))
    mouth_h = mouth_open_height(gray, mouth_box)
    smile = smile_score(gray, face_box)

    return {
        "eye_score": eye_score,
        "brow_row_frac": brow_row_frac,
        "mouth_h": mouth_h,
        "mouth_box_h": mouth_box[3],
        "smile": smile,
    }


def to_observation(raw: dict, baseline: Baseline, face_box: Box) -> Observation:
    eye_open = clamp(raw["eye_score"] / max(0.15, baseline.eye_score))
    brow_raise = clamp((baseline.brow_row_frac - raw["brow_row_frac"]) / 0.5)
    mouth_open = clamp((raw["mouth_h"] - baseline.mouth_h) / (raw["mouth_box_h"] * 0.35 + 1e-6))
    smile = clamp(raw["smile"])
    return Observation(box=face_box, eye_open=eye_open, brow_raise=brow_raise,
                        mouth_open=mouth_open, smile=smile)

def draw_debug(frame: np.ndarray, state: str, observation: Optional[Observation]) -> None:
    if observation is not None:
        x, y, w, h = observation.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
        lines = [
            "face: 1",
            f"eyeOpen: {observation.eye_open:.2f}",
            f"browRaise: {observation.brow_raise:.2f}",
            f"mouthOpen: {observation.mouth_open:.2f}",
            f"smile: {observation.smile:.2f}",
        ]
    else:
        lines = ["face: 0", "eyeOpen: --", "browRaise: --", "mouthOpen: --", "smile: --"]

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + i * 20), FONT, 0.55, BOX_COLOR, 1, cv2.LINE_AA)

    cv2.putText(frame, f"state: {state}", (10, frame.shape[0] - 12), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

def _bezier(p0, p1, p2, p3, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] +
             3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0])
        y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] +
             3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1])
        pts.append((x, y))
    return pts


def _draw_head_outline(panel, cx, cy, head_w, head_h):
    line_w = max(2, int(head_w * 0.02))
    top = cy - head_h * 0.62
    bottom = cy + head_h * 0.58
    p_bl = (cx - head_w * 0.02, bottom)
    p_lm = (cx - head_w * 0.40, top + head_h * 0.08)
    p_rm = (cx + head_w * 0.40, top + head_h * 0.08)
    p_br = (cx + head_w * 0.02, bottom)
    pts = []
    pts += _bezier(p_bl, (cx - head_w * 0.62, cy + head_h * 0.30),
                    (cx - head_w * 0.66, cy - head_h * 0.25), p_lm)
    pts += _bezier(p_lm, (cx - head_w * 0.18, top - head_h * 0.03),
                    (cx + head_w * 0.18, top - head_h * 0.03), p_rm)
    pts += _bezier(p_rm, (cx + head_w * 0.66, cy - head_h * 0.25),
                    (cx + head_w * 0.62, cy + head_h * 0.30), p_br)
    cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, INK, line_w, cv2.LINE_AA)
    return top, bottom


def _scribble_eye(panel, ex, ey, r):
    """Small hatched-circle eye: a generic sketchy-doodle convention, built
    from a plain ring plus a few short random chords — not any one
    character's specific eye design."""
    rng = np.random.default_rng(int(ex * 7 + ey * 13) & 0xFFFF)
    cv2.circle(panel, (int(ex), int(ey)), max(1, int(r)), INK, max(1, int(r * 0.35)), cv2.LINE_AA)
    for _ in range(5):
        a1 = rng.uniform(0, 2 * np.pi)
        a2 = a1 + rng.uniform(0.6, 1.4)
        p1 = (int(ex + np.cos(a1) * r * 0.7), int(ey + np.sin(a1) * r * 0.7))
        p2 = (int(ex + np.cos(a2) * r * 0.7), int(ey + np.sin(a2) * r * 0.7))
        cv2.line(panel, p1, p2, INK, 1, cv2.LINE_AA)


def _draw_nose(panel, cx, cy, head_w, head_h):
    cv2.ellipse(panel, (int(cx), int(cy + head_h * 0.14)),
                (int(head_w * 0.045), int(head_h * 0.03)), 0, 0, 360, NOSE_COLOR, -1, cv2.LINE_AA)


def _pose_neutral(panel, cx, cy, head_w, head_h):
    _draw_head_outline(panel, cx, cy, head_w, head_h)
    ey = cy - head_h * 0.02
    _scribble_eye(panel, cx - head_w * 0.24, ey, head_w * 0.05)
    _scribble_eye(panel, cx + head_w * 0.24, ey, head_w * 0.05)
    _draw_nose(panel, cx, cy, head_w, head_h)
    mouth_y = cy + head_h * 0.34
    curve = _bezier((cx - head_w * 0.14, mouth_y - 3), (cx, mouth_y + 3), (cx, mouth_y + 3),
                     (cx + head_w * 0.14, mouth_y - 3), n=16)
    cv2.polylines(panel, [np.array(curve, dtype=np.int32)], False, INK, max(2, int(head_w * 0.02)), cv2.LINE_AA)


def _pose_blink(panel, cx, cy, head_w, head_h):
    _draw_head_outline(panel, cx, cy, head_w, head_h)
    ey = cy - head_h * 0.02
    for side in (-1, 1):
        ex = cx + side * head_w * 0.24
        curve = _bezier((ex - head_w * 0.06, ey), (ex, ey + head_h * 0.02), (ex, ey + head_h * 0.02),
                         (ex + head_w * 0.06, ey), n=12)
        cv2.polylines(panel, [np.array(curve, dtype=np.int32)], False, INK,
                      max(2, int(head_w * 0.022)), cv2.LINE_AA)
    _draw_nose(panel, cx, cy, head_w, head_h)
    mouth_y = cy + head_h * 0.34
    curve = _bezier((cx - head_w * 0.13, mouth_y - 2), (cx, mouth_y + 4), (cx, mouth_y + 4),
                     (cx + head_w * 0.13, mouth_y - 2), n=16)
    cv2.polylines(panel, [np.array(curve, dtype=np.int32)], False, INK, max(2, int(head_w * 0.02)), cv2.LINE_AA)


def _pose_smile(panel, cx, cy, head_w, head_h):
    _draw_head_outline(panel, cx, cy, head_w, head_h)
    ey = cy - head_h * 0.03
    for side in (-1, 1):
        ex = cx + side * head_w * 0.24
        curve = _bezier((ex - head_w * 0.055, ey + head_h * 0.02), (ex, ey - head_h * 0.02),
                         (ex, ey - head_h * 0.02), (ex + head_w * 0.055, ey + head_h * 0.02), n=12)
        cv2.polylines(panel, [np.array(curve, dtype=np.int32)], False, INK,
                      max(2, int(head_w * 0.022)), cv2.LINE_AA)
    _draw_nose(panel, cx, cy, head_w, head_h)
    mouth_y = cy + head_h * 0.32
    curve = _bezier((cx - head_w * 0.20, mouth_y - 2), (cx, mouth_y + head_h * 0.08),
                     (cx, mouth_y + head_h * 0.08), (cx + head_w * 0.20, mouth_y - 2), n=20)
    cv2.polylines(panel, [np.array(curve, dtype=np.int32)], False, INK, max(2, int(head_w * 0.025)), cv2.LINE_AA)


def _pose_surprised(panel, cx, cy, head_w, head_h):
    _draw_head_outline(panel, cx, cy, head_w, head_h)
    brow_y = cy - head_h * 0.24
    for side in (-1, 1):
        bx = cx + side * head_w * 0.24
        curve = _bezier((bx - head_w * 0.08, brow_y + 4), (bx, brow_y - 6), (bx, brow_y - 6),
                         (bx + head_w * 0.08, brow_y + 4), n=14)
        cv2.polylines(panel, [np.array(curve, dtype=np.int32)], False, INK,
                      max(2, int(head_w * 0.03)), cv2.LINE_AA)
    ey = cy - head_h * 0.02
    for side in (-1, 1):
        ex = cx + side * head_w * 0.24
        cv2.ellipse(panel, (int(ex), int(ey)), (int(head_w * 0.05), int(head_h * 0.065)),
                    0, 0, 360, INK, -1, cv2.LINE_AA)
    _draw_nose(panel, cx, cy, head_w, head_h)
    mouth_y = cy + head_h * 0.35
    r = head_w * 0.10
    cv2.ellipse(panel, (int(cx), int(mouth_y)), (int(r), int(r * 1.2)), 0, 0, 360, INK,
                max(2, int(head_w * 0.02)), cv2.LINE_AA)


def _pose_tongue(panel, cx, cy, head_w, head_h):
    _draw_head_outline(panel, cx, cy, head_w, head_h)
    ey = cy - head_h * 0.02
    _scribble_eye(panel, cx - head_w * 0.24, ey, head_w * 0.05)
    _scribble_eye(panel, cx + head_w * 0.24, ey, head_w * 0.05)
    _draw_nose(panel, cx, cy, head_w, head_h)

    mouth_y = cy + head_h * 0.32
    mw, mh = head_w * 0.19, head_h * 0.075
    cavity_pts = [(cx - mw, mouth_y)]
    for t in range(0, 181, 10):
        a = np.radians(t)
        cavity_pts.append((cx + np.cos(a + np.pi) * mw, mouth_y + abs(np.sin(a)) * mh * 1.7))
    cavity_pts.append((cx + mw, mouth_y))
    cv2.fillPoly(panel, [np.array(cavity_pts, dtype=np.int32)], INK, cv2.LINE_AA)
    cv2.line(panel, (int(cx - mw), int(mouth_y)), (int(cx + mw), int(mouth_y)), INK, 2, cv2.LINE_AA)

    tongue_cy = mouth_y + mh * 1.35
    cv2.ellipse(panel, (int(cx), int(tongue_cy)), (int(mw * 0.5), int(mh * 1.0)),
                0, 0, 180, TONGUE_COLOR, -1, cv2.LINE_AA)
    cv2.line(panel, (int(cx), int(tongue_cy)), (int(cx), int(tongue_cy - mh * 0.5)),
             (140, 110, 200), 1, cv2.LINE_AA)


POSES = {
    "neutral": _pose_neutral,
    "blink": _pose_blink,
    "smile": _pose_smile,
    "surprised": _pose_surprised,
    "tongue": _pose_tongue,
}


def classify(obs: Observation) -> str:
    """Map measured signals to one of the discrete poses above. Order
    matters: checked roughly most- to least-distinctive so one clear signal
    (eyes shut) isn't overridden by a noisier one (a borderline smile
    score)."""
    if obs.eye_open < 0.22:
        return "blink"
    if obs.brow_raise > 0.55 and obs.mouth_open > 0.35:
        return "surprised"
    if obs.mouth_open > 0.45:
        return "tongue"
    if obs.smile > 0.45:
        return "smile"
    return "neutral"


def draw_reaction(panel: np.ndarray, pose_name: str) -> None:
    h, w = panel.shape[:2]
    panel[:] = PANEL_BG
    cx, cy = w * 0.5, h * 0.52
    head_w = min(w, h) * 0.46
    head_h = min(w, h) * 0.62
    POSES.get(pose_name, _pose_neutral)(panel, cx, cy, head_w, head_h)


def draw_prompt(panel: np.ndarray, text: str) -> None:
    panel[:] = PANEL_BG
    size = cv2.getTextSize(text, FONT, 0.9, 2)[0]
    x = (panel.shape[1] - size[0]) // 2
    y = (panel.shape[0] + size[1]) // 2
    cv2.putText(panel, text, (x, y), FONT, 0.9, (75, 80, 71), 2, cv2.LINE_AA)

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera (index 0). If you have more than one "
              "camera, or another app is using it, try a different index.")
        return

    state = "waiting_calibration"
    baseline = Baseline()
    smooth_box: Optional[Box] = None
    m = {"eye_open": 0.75, "brow_raise": 0.0, "mouth_open": 0.0, "smile": 0.2}
    cal_samples = []

    # Debounce the discrete pose choice so it doesn't flicker between two
    # poses when a metric hovers right at a classification boundary: a
    # candidate has to win 3 frames in a row before it actually switches.
    current_pose = "neutral"
    pending_pose = "neutral"
    pending_count = 0
    DEBOUNCE_FRAMES = 3

    print("make a face — press 'c' to calibrate (hold a neutral, relaxed face), 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, like looking in a mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_box = detect_face(gray)
        obs: Optional[Observation] = None

        if raw_box is None:
            smooth_box = None
        else:
            if smooth_box is None:
                smooth_box = raw_box
            else:
                a = 0.35
                smooth_box = tuple(int(lerp(smooth_box[i], raw_box[i], a)) for i in range(4))

            raw = measure(gray, smooth_box)

            if state == "calibrating":
                cal_samples.append(raw)
                if len(cal_samples) >= CALIBRATION_FRAMES:
                    baseline = Baseline(
                        eye_score=max(0.15, np.mean([s["eye_score"] for s in cal_samples])),
                        brow_row_frac=float(np.mean([s["brow_row_frac"] for s in cal_samples])),
                        mouth_h=max(1.0, np.mean([s["mouth_h"] for s in cal_samples])),
                    )
                    state = "live"
                    cal_samples = []

            target = to_observation(raw, baseline, smooth_box)
            sm = 0.3
            m["eye_open"] = lerp(m["eye_open"], target.eye_open, sm)
            m["brow_raise"] = lerp(m["brow_raise"], target.brow_raise, sm)
            m["mouth_open"] = lerp(m["mouth_open"], target.mouth_open, sm)
            m["smile"] = lerp(m["smile"], target.smile, sm)
            obs = Observation(box=smooth_box, eye_open=m["eye_open"], brow_raise=m["brow_raise"],
                               mouth_open=m["mouth_open"], smile=m["smile"])

            if state == "live":
                candidate = classify(obs)
                if candidate == pending_pose:
                    pending_count += 1
                else:
                    pending_pose = candidate
                    pending_count = 1
                if pending_count >= DEBOUNCE_FRAMES:
                    current_pose = pending_pose

        cam_panel = cv2.resize(frame, (PANEL_W, PANEL_H))
        scale_x, scale_y = PANEL_W / frame.shape[1], PANEL_H / frame.shape[0]
        debug_obs = None
        if obs is not None:
            x, y, w, h = obs.box
            debug_obs = Observation(box=(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)),
                                     eye_open=obs.eye_open, brow_raise=obs.brow_raise,
                                     mouth_open=obs.mouth_open, smile=obs.smile)
        draw_debug(cam_panel, state, debug_obs)

        reaction_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        if state == "live" and obs is not None:
            draw_reaction(reaction_panel, current_pose)
        else:
            draw_prompt(reaction_panel, "make a face")

        combined = np.hstack([cam_panel, reaction_panel])
        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and state != "calibrating":
            state = "calibrating"
            cal_samples = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
