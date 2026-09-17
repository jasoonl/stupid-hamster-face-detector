from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import cv2

LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
LBF_MODEL_PATH = Path(__file__).resolve().parent / "lbfmodel.yaml"

BOOTSTRAP_DATASET = Path(__file__).resolve().parent / "features_bootstrap.npz"

USER_DATASET = Path(__file__).resolve().parent / "features_user.npz"

N_FEATURES = 12


def _download_with_ssl_fallback(url: str, dest_path: Path) -> None:
   
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

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)).astype(np.uint8)


def extract_features(bgr: np.ndarray, landmarks: np.ndarray,
                     face_box: Tuple[int, int, int, int]) -> np.ndarray:
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

    X_old, y_old = load_dataset(path)
    X = np.vstack([X_old, np.asarray(X_new, np.float32)]) if X_old.size else np.asarray(X_new, np.float32)
    y = np.concatenate([y_old, np.asarray(y_new, str)]) if y_old.size else np.asarray(y_new, str)
    np.savez(path, X=X, y=y)

class ExpressionClassifier:

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

        if self._clf is None:
            return None, 0.0
        proba = self._clf.predict_proba(feats.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return self._classes[i], float(proba[i])


def build_classifier() -> ExpressionClassifier:

    X, y = load_dataset(BOOTSTRAP_DATASET, USER_DATASET)
    clf = ExpressionClassifier()
    clf.train(X, y)
    return clf
