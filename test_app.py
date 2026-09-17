import tempfile
from pathlib import Path

import numpy as np
import cv2

import expression_engine as E
import app


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def synthetic_landmarks():

    pts = np.zeros((68, 2), dtype=np.float32)
    # eyes: 36 (left outer) .. 45 (right outer) -- set a real interocular span
    pts[36] = (200, 300); pts[45] = (360, 300)
    pts[37] = (215, 292); pts[41] = (215, 308)   # left eye top/bottom
    pts[43] = (345, 292); pts[47] = (345, 308)   # right eye top/bottom
    pts[19] = (215, 265); pts[24] = (345, 265)   # brows
    # mouth: outer 48 (L) / 54 (R), top 51, inner top 62 / inner bottom 66
    pts[48] = (250, 400); pts[54] = (310, 400)
    pts[51] = (280, 388)
    pts[62] = (280, 398); pts[66] = (280, 410)
    for i in range(48, 68):
        if np.all(pts[i] == 0):
            pts[i] = (280, 404)
    return pts


def main():
    results = []

    results.append(check("POSES has the 5 expected expressions",
                         set(E.POSES) == {"neutral", "shock", "tongue", "huh", "shush"}))
    results.append(check("N_FEATURES is a positive int", isinstance(E.N_FEATURES, int) and E.N_FEATURES > 0))

    img = np.full((480, 640, 3), (150, 130, 180), dtype=np.uint8)
    lm = synthetic_landmarks()
    box = (180, 240, 200, 200)
    feats = E.extract_features(img, lm, box)
    results.append(check(f"extract_features returns N_FEATURES ({E.N_FEATURES}) floats",
                         feats.shape == (E.N_FEATURES,) and feats.dtype == np.float32))

    feats_2x = E.extract_features(img, lm * 2, tuple(v * 2 for v in box))
    geo = slice(0, 6)
    results.append(check("geometry features are ~scale-invariant (2x face -> ~same values)",
                         np.allclose(feats[geo], feats_2x[geo], atol=0.05)))

    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "user.npz"
        Xa = np.random.RandomState(0).rand(3, E.N_FEATURES).astype(np.float32)
        E.append_samples(ds, Xa, np.array(["shock", "shock", "tongue"]))
        E.append_samples(ds, feats.reshape(1, -1), np.array(["neutral"]))
        X, y = E.load_dataset(ds)
        results.append(check("append_samples + load_dataset accumulate rows across calls",
                             X.shape == (4, E.N_FEATURES) and list(y) == ["shock", "shock", "tongue", "neutral"]))
        results.append(check("load_dataset on a missing path returns empty, no crash",
                             E.load_dataset(Path(tmp) / "nope.npz")[0].shape == (0, E.N_FEATURES)))

    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        have_sklearn = True
    except ImportError:
        have_sklearn = False

    if have_sklearn:
        rs = np.random.RandomState(1)
        Xa = np.vstack([rs.rand(20, E.N_FEATURES), rs.rand(20, E.N_FEATURES) + 5.0]).astype(np.float32)
        ya = np.array(["neutral"] * 20 + ["shock"] * 20)
        clf = E.ExpressionClassifier()
        results.append(check("classifier not ready before training", not clf.ready))
        clf.train(Xa, ya)
        results.append(check("classifier ready after training and reports n_train", clf.ready and clf.n_train == 40))
        pose_lo, conf_lo = clf.predict(np.zeros(E.N_FEATURES, np.float32))
        pose_hi, conf_hi = clf.predict(np.full(E.N_FEATURES, 5.0, np.float32))
        results.append(check("classifier separates the two clusters correctly",
                             pose_lo == "neutral" and pose_hi == "shock"))
        results.append(check("classifier confidence is a probability in [0,1]",
                             0.0 <= conf_lo <= 1.0 and 0.0 <= conf_hi <= 1.0))
        results.append(check("untrained classifier predict -> (None, 0.0)",
                             E.ExpressionClassifier().predict(feats) == (None, 0.0)))
        try:
            E.ExpressionClassifier().train(Xa[:1], ya[:1]); ok = False
        except ValueError:
            ok = True
        results.append(check("train refuses <2 samples / <2 classes", ok))
    else:
        print("SKIP  classifier tests (scikit-learn not installed)")

    solid = lambda w, h, c: np.full((h, w, 3), c, dtype=np.uint8)
    out = app.fit_image_to_panel(solid(200, 200, (10, 10, 200)), 400, 250)
    results.append(check("fit_image_to_panel: square into wide panel is padded with PANEL_BG",
                         tuple(int(v) for v in out[125, 5]) == app.PANEL_BG and
                         tuple(int(v) for v in out[125, 200]) == (10, 10, 200)))
    results.append(check("fit_image_to_panel: None returns a plain panel, no crash",
                         app.fit_image_to_panel(None, 100, 100).shape == (100, 100, 3)))
    # alpha compositing blends toward the background
    rgba = np.dstack([solid(200, 200, (10, 10, 200)), np.full((200, 200), 128, np.uint8)])
    center = tuple(int(v) for v in app.fit_image_to_panel(rgba, 300, 300)[150, 150])
    results.append(check("fit_image_to_panel: 50% alpha blends toward background",
                         center != (10, 10, 200) and center != app.PANEL_BG))

    idle = np.zeros((app.PANEL_H, app.PANEL_W, 3), np.uint8)
    app.draw_reaction_panel(idle, None)
    results.append(check("reaction panel idle: plain PANEL_BG background",
                         tuple(int(v) for v in idle[5, 5]) == app.PANEL_BG))
    results.append(check("reaction panel idle: 'make a face' text drawn in muted gray, not neon",
                         np.any(np.all(idle == np.array(app.IDLE_TEXT_COLOR, np.uint8), axis=-1)) and
                         not np.any(np.all(idle == np.array(app.NEON_GREEN, np.uint8), axis=-1))))
    matched = np.zeros((app.PANEL_H, app.PANEL_W, 3), np.uint8)
    app.draw_reaction_panel(matched, solid(100, 300, (200, 100, 50)))
    results.append(check("reaction panel matched: the photo's own color appears",
                         np.any(np.all(matched == np.array((200, 100, 50), np.uint8), axis=-1))))

    frame = np.zeros((app.PANEL_H, app.PANEL_W, 3), np.uint8)
    app.draw_debug(frame, (50, 50, 100, 100), "shock", 0.87,
                   {"mouth_open": 0.3, "brow_raise": 0.1, "eye_open": 0.2, "smile": 0.0},
                   {"neutral": 5, "shock": 3, "tongue": 4, "huh": 2, "shush": 6}, "recorded shock")
    results.append(check("draw_debug: draws neon-green overlay when a face is present",
                         np.any(np.all(frame == np.array(app.NEON_GREEN, np.uint8), axis=-1))))
    frame2 = np.zeros((app.PANEL_H, app.PANEL_W, 3), np.uint8)
    app.draw_debug(frame2, None, None, 0.0, None, {}, None)
    results.append(check("draw_debug: handles no-face / no-stats without crashing",
                         frame2.shape == (app.PANEL_H, app.PANEL_W, 3)))

    s = app.stats_from_features(feats)
    results.append(check("stats_from_features returns the 4 readout keys",
                         set(s.keys()) == {"mouth_open", "brow_raise", "eye_open", "smile"}))

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cv2.imwrite(str(d / "shock.jpg"), solid(60, 60, (0, 0, 255)))
        cv2.imwrite(str(d / "toungue.png"), solid(60, 60, (0, 255, 0)))   # misspelling alias
        imgs = app.load_reference_images(d)
        results.append(check("load_reference_images: finds shock.jpg", "shock" in imgs))
        results.append(check("load_reference_images: matches 'toungue' misspelling to tongue", "tongue" in imgs))
        results.append(check("load_reference_images: absent poses simply missing", "huh" not in imgs))
        results.append(check("load_reference_images: missing dir -> empty dict, no crash",
                             app.load_reference_images(d / "nope") == {}))

    print()
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
