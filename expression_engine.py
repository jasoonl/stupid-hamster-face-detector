"""
expression_engine.py -- the recognition brain for "make a face".

This replaces the old pixel/template-matching approach, which was proven
(against real webcam footage) to be the wrong tool: it compared grayscale
mouth crops and could not tell a neutral face from a shocked one -- even
when the crops were perfectly aligned it scored ~3/13, because a
histogram-equalized mouth crop looks structurally similar no matter the
expression. Worse, the old code located the "mouth" as a fixed proportion
of the OpenCV Haar face box, and that box's vertical extent swings wildly
frame to frame (measured: 149px vs 259px on the same face), so the crop
routinely landed on the nose or chin instead of the mouth.

The approach here instead:

  1. Track 68 facial landmarks with OpenCV's LBF facemark
     (cv2.face.createFacemarkLBF) -- a pure-CPU model, so it CANNOT hit
     the MediaPipe Metal-GPU crash that aborted the old app on macOS.
     Landmarks anchor on the real mouth every frame, regardless of how
     the coarse face box jitters.

  2. Turn those landmarks (+ a color read of the mouth interior + skin
     coverage in zones around the face for the hand gestures) into a
     small, scale-invariant FEATURE VECTOR describing the expression.

  3. Classify that vector with a RandomForest TRAINED ON REAL LABELED
     FRAMES. On the user's own footage this separated all five states
     (neutral / shock / tongue / huh / shush) at ~94% five-fold accuracy,
     versus the old matcher's ~25%. The model is trained at runtime from
     a shipped dataset of feature *numbers* (features_bootstrap.npz) plus
     any samples the user records in-app -- never from a pickled model
     object, so there's nothing to break across scikit-learn versions.

Everything here is camera-independent and unit-testable: feature
extraction takes a BGR image, the classifier takes feature arrays.

Requires: opencv-contrib-python (for cv2.face), numpy, scikit-learn.
"""
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import cv2


# The 5 expressions, in a fixed canonical order (indices are stable so a
# saved dataset stays readable if this file changes).
POSES = ["neutral", "shock", "tongue", "huh", "shush"]

# Which poses are hand gestures (used only for messaging / optional
# weighting -- the classifier itself treats all 5 uniformly).
HAND_POSES = {"huh", "shush"}

# OpenCV's LBF 68-point facial-landmark model. Pure CPU. Downloaded once
# (like the old hand model was) and cached next to this file.
LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
LBF_MODEL_PATH = Path(__file__).resolve().parent / "lbfmodel.yaml"

# Shipped bootstrap training data: feature vectors + labels extracted from
# the reference footage, so the app recognizes faces out of the box before
# the user records anything of their own. Just numbers -- no images.
BOOTSTRAP_DATASET = Path(__file__).resolve().parent / "features_bootstrap.npz"
# Where the user's own recorded samples accumulate (grows accuracy over time).
USER_DATASET = Path(__file__).resolve().parent / "features_user.npz"

# Number of features produced by extract_features(); asserted at runtime so
# a mismatch between a saved dataset and the current extractor is caught
# loudly instead of silently corrupting training.
N_FEATURES = 12


def _download_with_ssl_fallback(url: str, dest_path: Path) -> None:
    """Download `url` to `dest_path`, working around the common macOS
    python.org "CERTIFICATE_VERIFY_FAILED" issue: tries the default SSL
    context, then the certifi CA bundle if installed, then (last resort,
    for this fixed public model file only) an unverified connection.
    Raises on total failure."""
    import ssl
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlretrieve(url, str(dest_path))
        return
    except urllib.error.URLError:
        pass

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        return
    except ImportError:
        pass
    except Exception:
        pass

    print("  Your Python installation can't verify HTTPS certificates (common after installing "
          "Python from python.org on macOS). Falling back to an unverified connection for this "
          "one download only -- it's a fixed, public model file. To fix it properly: "
          "`pip3 install certifi`, or run \"Install Certificates.command\" in your Python folder.")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


class LandmarkTracker:
    """Wraps a Haar face detector + LBF 68-point facemark into one call:
    image -> (landmarks[68,2], face_box) or (None, None). Pure CPU."""

    def __init__(self, model_path: Path = LBF_MODEL_PATH):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face is missing -- you have plain opencv-python, but the landmark tracker "
                "needs opencv-contrib-python. Fix: pip3 uninstall -y opencv-python && "
                "pip3 install 'opencv-contrib-python<5'")
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not model_path.exists():
            print(f"Downloading the face-landmark model once (~54MB, saves to {model_path.name})...")
            _download_with_ssl_fallback(LBF_MODEL_URL, model_path)
        self._fm = cv2.face.createFacemarkLBF()
        self._fm.loadModel(str(model_path))

    def fit(self, bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None, None
        # largest face only
        faces = np.array(sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1])
        ok, landmarks = self._fm.fit(gray, faces)
        if not ok:
            return None, None
        return landmarks[0][0], tuple(int(v) for v in faces[0])


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary skin mask in YCrCb -- robust across skin tones and lighting,
    the standard cheap skin detector. Used only for the hand-gesture
    features (how much skin sits in zones around the face)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)).astype(np.uint8)


def extract_features(bgr: np.ndarray, landmarks: np.ndarray,
                     face_box: Tuple[int, int, int, int]) -> np.ndarray:
    """Turn one face's 68 landmarks (+ the image, for mouth color and hand
    skin coverage) into the fixed-length feature vector the classifier
    uses. Every geometric feature is normalized by the interocular
    distance so it's invariant to how close/far the face is from the
    camera. Returns a float32 array of length N_FEATURES.

    Feature groups:
      geometry (6): mouth open gap, mouth width, openness/width ratio,
                    brow raise, eye openness, smile (corner-vs-top-lip).
      mouth color (2): dark-cavity fraction (open mouth), pink/red
                    fraction (tongue) inside a mouth-centered ROI.
      hand zones (4): skin coverage below / directly-over-mouth / left /
                    right of the face -- what separates the hand gestures
                    (huh = hands spread below; shush = finger over mouth)
                    from face-only expressions.
    """
    pts = landmarks
    x, y, w, h = face_box
    H, W = bgr.shape[:2]
    io = float(np.linalg.norm(pts[36] - pts[45])) + 1e-6   # interocular distance
    d = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))

    geom = [
        d(62, 66) / io,                                     # inner-lip vertical gap (openness)
        d(48, 54) / io,                                     # mouth width
        d(62, 66) / (d(48, 54) + 1e-6),                     # openness relative to width
        (pts[[19, 24], 1].mean() - pts[[37, 44], 1].mean()) / io * -1.0,  # brow raise
        (d(37, 41) + d(43, 47)) / 2 / io,                   # eye openness
        (pts[[48, 54], 1].mean() - pts[51, 1]) / io,        # mouth corners vs top lip (smile)
    ]

    # mouth-interior color ROI, centered on the mouth landmarks and sized
    # by interocular distance (stable) rather than the jittery face box.
    mc = pts[48:68].mean(0)
    r = 0.55 * io
    rx0, ry0 = int(mc[0] - r), int(mc[1] - 0.4 * io)
    rx1, ry1 = int(mc[0] + r), int(mc[1] + 0.8 * io)
    rx0, ry0, rx1, ry1 = max(0, rx0), max(0, ry0), min(W, rx1), min(H, ry1)
    roi = bgr[ry0:ry1, rx0:rx1]
    if roi.size:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
        hh, ss, vv = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        dark = float((vv < max(30, np.median(vv) * 0.5)).mean())
        pink = float((((hh < 12) | (hh > 168)) & (ss > 90) & (vv > 70) & (vv < 210)).mean())
    else:
        dark = pink = 0.0

    sm = _skin_mask(bgr)

    def zone_frac(a, b, c, dd):
        a, b, c, dd = max(0, a), max(0, b), min(W, c), min(H, dd)
        z = sm[b:dd, a:c]
        return float(z.mean()) if z.size else 0.0

    hands = [
        zone_frac(x - w // 2, y + h, x + w + w // 2, H),               # below face (huh)
        zone_frac(x + w // 4, y + int(0.55 * h), x + 3 * w // 4, y + h + h // 3),  # over mouth (shush)
        zone_frac(0, y, x, y + h + h // 2),                            # left of face
        zone_frac(x + w, y, W, y + h + h // 2),                        # right of face
    ]

    feats = np.array(geom + [dark, pink] + hands, dtype=np.float32)
    assert feats.shape[0] == N_FEATURES, f"expected {N_FEATURES} features, got {feats.shape[0]}"
    return feats


# ----------------------------------------------------------------------------
# dataset IO (feature vectors + string labels; just numbers, never images)
# ----------------------------------------------------------------------------

def load_dataset(*paths: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate one or more .npz datasets (each with arrays
    'X' [n, N_FEATURES] and 'y' [n] of pose strings). Missing files are
    skipped. Returns (X, y); empty arrays if nothing loads."""
    Xs, ys = [], []
    for p in paths:
        if p and Path(p).exists():
            data = np.load(p, allow_pickle=True)
            X, y = data["X"], data["y"]
            if X.size and X.shape[1] == N_FEATURES:
                Xs.append(X.astype(np.float32))
                ys.append(y.astype(str))
    if not Xs:
        return np.empty((0, N_FEATURES), np.float32), np.empty((0,), str)
    return np.vstack(Xs), np.concatenate(ys)


def append_samples(path: Path, X_new: np.ndarray, y_new: np.ndarray) -> None:
    """Append feature rows + labels to a user dataset (creating it if
    needed), so recorded samples accumulate across sessions."""
    X_old, y_old = load_dataset(path)
    X = np.vstack([X_old, np.asarray(X_new, np.float32)]) if X_old.size else np.asarray(X_new, np.float32)
    y = np.concatenate([y_old, np.asarray(y_new, str)]) if y_old.size else np.asarray(y_new, str)
    np.savez(path, X=X, y=y)


# ----------------------------------------------------------------------------
# classifier (trained at runtime from the datasets above)
# ----------------------------------------------------------------------------

class ExpressionClassifier:
    """A RandomForest over expression feature vectors, trained in-process
    from the shipped bootstrap dataset plus any user-recorded samples.
    Trained at load time (fast: <1s for ~100 samples) so there's no
    pickled model to break across scikit-learn versions."""

    def __init__(self):
        self._clf = None
        self._classes: List[str] = []
        self._n_train = 0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestClassifier
        if X.shape[0] < 2 or len(set(y)) < 2:
            raise ValueError("need at least 2 samples spanning at least 2 poses to train")
        clf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
        clf.fit(X, y)
        self._clf = clf
        self._classes = list(clf.classes_)
        self._n_train = X.shape[0]

    @property
    def ready(self) -> bool:
        return self._clf is not None

    @property
    def n_train(self) -> int:
        return self._n_train

    def predict(self, feats: np.ndarray) -> Tuple[Optional[str], float]:
        """Return (pose, confidence). confidence is the winning class
        probability in [0,1]. Returns (None, 0.0) if untrained."""
        if self._clf is None:
            return None, 0.0
        proba = self._clf.predict_proba(feats.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return self._classes[i], float(proba[i])


def build_classifier() -> ExpressionClassifier:
    """Load bootstrap + user datasets and train. Raises if no data at all
    (so the caller can fall back to a clear message)."""
    X, y = load_dataset(BOOTSTRAP_DATASET, USER_DATASET)
    clf = ExpressionClassifier()
    clf.train(X, y)
    return clf
