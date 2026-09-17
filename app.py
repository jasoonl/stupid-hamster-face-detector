from pathlib import Path
from typing import Optional, Tuple, Dict
import sys
import time

import cv2
import numpy as np

import expression_engine as engine


# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

WINDOW_NAME = "make a face"
NEON_GREEN = (20, 255, 57)       # BGR -- the #39FF14 neon-green look from the reference video
PANEL_BG = (234, 242, 244)       # plain near-white reaction panel, matching the video
FONT = cv2.FONT_HERSHEY_SIMPLEX
IDLE_FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX      # soft handwritten look for the idle prompt
IDLE_TEXT_COLOR = (150, 150, 150)

PANEL_W, PANEL_H = 480, 480      # each side of the combined window
CAMERA_INDEX = 0

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
IMAGE_ALIASES = {
    "neutral": ["neutral", "blank", "default"],
    "shock": ["shock", "shocked", "surprised"],
    "tongue": ["tongue", "toungue"],
    "shush": ["shush"],
    "huh": ["huh", "shrug"],
}

# Confidence the classifier must reach before we actually show a pose's
# photo. Below this we show the idle "make a face" prompt rather than
# guessing. Tunable; 0.55 rejected the borderline frames in testing while
# still firing on clear expressions.
CONFIDENCE_THRESHOLD = 0.55
# A pose must win this many frames in a row before the shown photo
# switches -- debounce, so it doesn't flicker between two near-tied poses.
HOLD_FRAMES = 3
# "neutral" is a real class the model predicts, but it isn't a photo we
# pop up as a "reaction" -- when neutral wins, we show the idle prompt.
IDLE_POSE = "neutral"

# number-key -> pose, for the in-app "teach it" data collector
RECORD_KEYS = {ord("1"): "neutral", ord("2"): "shock", ord("3"): "tongue",
               ord("4"): "huh", ord("5"): "shush"}


# ----------------------------------------------------------------------------
# reference images + UI (ported unchanged from the previous version)
# ----------------------------------------------------------------------------

def fit_image_to_panel(img: Optional[np.ndarray], w: int, h: int) -> np.ndarray:
    """Resize `img` to fit inside a w x h panel, preserving aspect ratio,
    centered on PANEL_BG, with alpha compositing if present. Never
    distorts the source proportions."""
    panel = np.full((h, w, 3), PANEL_BG, dtype=np.uint8)
    if img is None or img.size == 0:
        return panel

    has_alpha = img.ndim == 3 and img.shape[2] == 4
    src_h, src_w = img.shape[:2]
    scale = min(w / src_w, h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0, y0 = (w - new_w) // 2, (h - new_h) // 2

    if has_alpha:
        bgr = resized[:, :, :3].astype(np.float32)
        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        region = panel[y0:y0 + new_h, x0:x0 + new_w].astype(np.float32)
        panel[y0:y0 + new_h, x0:x0 + new_w] = (bgr * alpha + region * (1 - alpha)).astype(np.uint8)
    else:
        panel[y0:y0 + new_h, x0:x0 + new_w] = resized
    return panel


def load_reference_images(images_dir: Path) -> Dict[str, np.ndarray]:
    """Return {pose_name: raw image} for every reference photo found.
    Stored raw (unfit) so draw_reaction_panel fits exactly once at render
    time (fitting twice baked a visible letterbox bar into non-square
    photos in the old version)."""
    display_images: Dict[str, np.ndarray] = {}
    if not images_dir.is_dir():
        return display_images
    available = {p.name.lower(): p for p in images_dir.iterdir() if p.is_file()}
    for pose_name, aliases in IMAGE_ALIASES.items():
        for alias in aliases:
            hit = next((available[(alias + ext).lower()] for ext in IMAGE_EXTENSIONS
                        if (alias + ext).lower() in available), None)
            if hit is not None:
                raw = cv2.imread(str(hit), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    display_images[pose_name] = raw
                break
    return display_images


def draw_reaction_panel(panel: np.ndarray, image: Optional[np.ndarray]) -> None:
    """Right panel, matching the reference video: a plain light background
    with the matched photo shown large (small margin), or a soft
    handwritten "make a face" prompt when nothing is matched."""
    h, w = panel.shape[:2]
    panel[:] = PANEL_BG
    if image is not None:
        margin = 18
        inset = fit_image_to_panel(image, w - 2 * margin, h - 2 * margin)
        panel[margin:margin + inset.shape[0], margin:margin + inset.shape[1]] = inset
    else:
        text = "make a face"
        size = cv2.getTextSize(text, IDLE_FONT, 1.3, 2)[0]
        cv2.putText(panel, text, ((w - size[0]) // 2, (h + size[1]) // 2),
                    IDLE_FONT, 1.3, IDLE_TEXT_COLOR, 2, cv2.LINE_AA)


def draw_debug(frame: np.ndarray, face_box: Optional[Tuple[int, int, int, int]],
               pred: Optional[str], conf: float, stats: Optional[dict],
               sample_counts: Dict[str, int], toast: Optional[str]) -> None:
    """Left panel overlay, styled like the video: a plain neon-green box,
    small left-aligned neon lines (expression + confidence + a few real
    landmark stats), a bottom legend for the teach-it keys, and a brief
    toast when a sample is recorded."""
    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), NEON_GREEN, 2)

    lines = [f"expression: {pred}" if pred else "expression: -",
             f"confidence: {conf:.2f}"]
    if stats is not None:
        lines += [f"mouthOpen: {stats['mouth_open']:.2f}",
                  f"browRaise: {stats['brow_raise']:.2f}",
                  f"eyeOpen: {stats['eye_open']:.2f}",
                  f"smile: {stats['smile']:.2f}"]
    else:
        lines.append("no face detected")
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + 22 * i), FONT, 0.55, NEON_GREEN, 1, cv2.LINE_AA)

    # bottom: teach-it legend + how many samples the model has of each pose
    H = frame.shape[0]
    legend = "teach: 1 neutral  2 shock  3 tongue  4 huh  5 shush   (q quit)"
    cv2.putText(frame, legend, (10, H - 34), FONT, 0.44, NEON_GREEN, 1, cv2.LINE_AA)
    counts = "  ".join(f"{p[:3]}:{sample_counts.get(p, 0)}" for p in engine.POSES)
    cv2.putText(frame, "samples  " + counts, (10, H - 14), FONT, 0.44, NEON_GREEN, 1, cv2.LINE_AA)

    if toast:
        size = cv2.getTextSize(toast, FONT, 0.7, 2)[0]
        cv2.putText(frame, toast, ((frame.shape[1] - size[0]) // 2, H - 60),
                    FONT, 0.7, NEON_GREEN, 2, cv2.LINE_AA)


def stats_from_features(feats: np.ndarray) -> dict:
    """Human-readable landmark stats for the readout (real values now,
    unlike the old broken heuristics). Indices match extract_features()."""
    return {"mouth_open": float(feats[2]), "brow_raise": float(feats[3]),
            "eye_open": float(feats[4]), "smile": float(feats[5])}


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

def main() -> None:
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX

    # --- recognition brain (CPU landmarks + trained classifier) ---
    try:
        tracker = engine.LandmarkTracker()
    except Exception as e:
        print(f"Couldn't start the face-landmark tracker: {e}")
        print("Make sure you have opencv-contrib-python (not plain opencv-python) and a network "
              "connection for the one-time model download.")
        return
    try:
        classifier = engine.build_classifier()
        print(f"Recognition model ready ({classifier.n_train} training samples). "
              f"Press 1-5 while making a face to teach it yours.")
    except Exception as e:
        classifier = engine.ExpressionClassifier()
        print(f"No training data yet ({e}). Press 1-5 while making each face to teach it, "
              f"then it'll start recognizing.")

    def current_counts() -> Dict[str, int]:
        X, y = engine.load_dataset(engine.BOOTSTRAP_DATASET, engine.USER_DATASET)
        return {p: int((y == p).sum()) for p in engine.POSES} if y.size else {p: 0 for p in engine.POSES}

    sample_counts = current_counts()

    display_images = load_reference_images(IMAGES_DIR)
    if display_images:
        print(f"Loaded reference photos: {sorted(display_images)}")
    else:
        print(f"No images found in {IMAGES_DIR} -- add shock/tongue/huh/shush/neutral photos there "
              f"so there's something to show when a face is recognized.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}. If you have more than one camera "
              f"(e.g. an iPhone Continuity Camera taking over index 0), try: python3 app.py 1")
        return
    print(f"Camera {camera_index} open. Wrong camera? Quit and run: python3 app.py 1")

    pending_pose: Optional[str] = None
    pending_count = 0
    shown_pose: Optional[str] = None
    toast_text: Optional[str] = None
    toast_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror

        landmarks, face_box = tracker.fit(frame)
        pred, conf, stats, feats = None, 0.0, None, None
        if landmarks is not None:
            feats = engine.extract_features(frame, landmarks, face_box)
            stats = stats_from_features(feats)
            if classifier.ready:
                pred, conf = classifier.predict(feats)

        # debounce the winning pose
        winner = pred if (pred is not None and conf >= CONFIDENCE_THRESHOLD) else None
        if winner == pending_pose:
            pending_count += 1
        else:
            pending_pose, pending_count = winner, 1
        if pending_count >= HOLD_FRAMES:
            shown_pose = pending_pose

        # which photo (if any) to show: the matched pose's image, unless
        # it's neutral / unknown / has no photo -> idle prompt
        show_image = None
        if shown_pose is not None and shown_pose != IDLE_POSE:
            show_image = display_images.get(shown_pose)

        # --- render ---
        cam_panel = cv2.resize(frame, (PANEL_W, PANEL_H))
        scaled_box = None
        if face_box is not None:
            sx, sy = PANEL_W / frame.shape[1], PANEL_H / frame.shape[0]
            x, y, w, h = face_box
            scaled_box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
        toast = toast_text if time.monotonic() < toast_until else None
        disp_pred = shown_pose if (shown_pose is not None) else pred
        draw_debug(cam_panel, scaled_box, disp_pred, conf, stats, sample_counts, toast)

        reaction_panel = np.empty((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        draw_reaction_panel(reaction_panel, show_image)

        cv2.imshow(WINDOW_NAME, np.hstack([cam_panel, reaction_panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in RECORD_KEYS:
            pose = RECORD_KEYS[key]
            if feats is None:
                toast_text, toast_until = "no face -- can't record", time.monotonic() + 1.2
            else:
                engine.append_samples(engine.USER_DATASET, feats.reshape(1, -1), np.array([pose]))
                sample_counts = current_counts()
                try:
                    classifier = engine.build_classifier()
                    toast_text = f"recorded {pose}  (retrained: {classifier.n_train} samples)"
                except Exception:
                    toast_text = f"recorded {pose}"
                toast_until = time.monotonic() + 1.4
                print(f"Recorded a '{pose}' sample. Model now has {classifier.n_train} samples.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
