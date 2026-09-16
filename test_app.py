import numpy as np
import app


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"{status}  {name}")
    return cond


def main():
    results = []

    results.append(check("face cascade loaded", not app.FACE_CASCADE.empty()))
    results.append(check("eye cascade loaded", not app.EYE_CASCADE.empty()))
    results.append(check("smile cascade loaded", not app.SMILE_CASCADE.empty()))

    results.append(check("clamp: below range", app.clamp(-1, 0, 1) == 0))
    results.append(check("clamp: above range", app.clamp(2, 0, 1) == 1))
    results.append(check("clamp: inside range", app.clamp(0.4, 0, 1) == 0.4))
    results.append(check("lerp: halfway", app.lerp(0, 10, 0.5) == 5))
    results.append(check("lerp: t=0 returns a", app.lerp(3, 9, 0) == 3))
    results.append(check("lerp: t=1 returns b", app.lerp(3, 9, 1) == 9))
  
    flat = np.array([50, 50, 50, 50, 50], dtype=float)
    results.append(check("dip width: flat profile has some width", app._contiguous_dip_width(flat) >= 1))

    narrow_dip = np.array([50, 50, 10, 50, 50], dtype=float)
    wide_dip = np.array([50, 20, 10, 20, 50], dtype=float)
    w_narrow = app._contiguous_dip_width(narrow_dip)
    w_wide = app._contiguous_dip_width(wide_dip)
    results.append(check(f"dip width: wider dip measured wider ({w_narrow} -> {w_wide})", w_wide > w_narrow))

    off_center = np.array([50, 50, 50, 10, 50, 50, 50], dtype=float)
    results.append(check("dip width: off-center dip still found narrow",
                          app._contiguous_dip_width(off_center) <= 3))

    baseline = app.Baseline(eye_score=1.0, brow_row_frac=0.5, mouth_h=5.0)
    box = (0, 0, 100, 100)

    neutral_raw = {"eye_score": 1.0, "brow_row_frac": 0.5, "mouth_h": 5.0, "mouth_box_h": 30, "smile": 0.0}
    obs = app.to_observation(neutral_raw, baseline, box)
    results.append(check("observation: neutral-vs-own-baseline is all near zero/high-open",
                          obs.eye_open > 0.9 and obs.brow_raise < 0.05 and obs.mouth_open < 0.05 and obs.smile < 0.05))

    open_mouth_raw = dict(neutral_raw, mouth_h=16.0)
    obs2 = app.to_observation(open_mouth_raw, baseline, box)
    results.append(check("observation: taller mouth dip -> higher mouth_open", obs2.mouth_open > obs.mouth_open))

    raised_brow_raw = dict(neutral_raw, brow_row_frac=0.2)
    obs3 = app.to_observation(raised_brow_raw, baseline, box)
    results.append(check("observation: brow row moving up -> higher brow_raise", obs3.brow_raise > obs.brow_raise))

    closed_eye_raw = dict(neutral_raw, eye_score=0.1)
    obs4 = app.to_observation(closed_eye_raw, baseline, box)
    results.append(check("observation: lower eye score -> lower eye_open", obs4.eye_open < obs.eye_open))

    for o in (obs, obs2, obs3, obs4):
        for field in ("eye_open", "brow_raise", "mouth_open", "smile"):
            v = getattr(o, field)
            if not (0.0 <= v <= 1.0):
                results.append(check(f"observation: {field} in [0,1] (got {v})", False))

    def obs(eye_open=0.8, brow_raise=0.0, mouth_open=0.0, smile=0.0):
        return app.Observation(box=(0, 0, 10, 10), eye_open=eye_open, brow_raise=brow_raise,
                                mouth_open=mouth_open, smile=smile)

    results.append(check("classify: relaxed neutral face -> neutral", app.classify(obs()) == "neutral"))
    results.append(check("classify: eyes shut -> blink", app.classify(obs(eye_open=0.05)) == "blink"))
    results.append(check("classify: wide smile -> smile", app.classify(obs(smile=0.8)) == "smile"))
    results.append(check("classify: open mouth alone -> tongue", app.classify(obs(mouth_open=0.7)) == "tongue"))
    results.append(check("classify: raised brows + open mouth -> surprised",
                          app.classify(obs(brow_raise=0.8, mouth_open=0.6)) == "surprised"))
    results.append(check("classify: closed eyes wins over an incidentally-high smile score",
                          app.classify(obs(eye_open=0.05, smile=0.9)) == "blink"))
    results.append(check("classify: every pose name is a real, drawable pose",
                          all(name in app.POSES for name in
                              (app.classify(obs()), app.classify(obs(eye_open=0.05)),
                               app.classify(obs(smile=0.8)), app.classify(obs(mouth_open=0.7)),
                               app.classify(obs(brow_raise=0.8, mouth_open=0.6))))))

    for name in app.POSES:
        panel = np.zeros((200, 200, 3), dtype=np.uint8)
        app.draw_reaction(panel, name)
        non_bg = np.any(panel != np.array(app.PANEL_BG, dtype=np.uint8))
        results.append(check(f"pose '{name}' renders and draws something", non_bg))

    panel = np.zeros((200, 200, 3), dtype=np.uint8)
    try:
        app.draw_reaction(panel, "not_a_real_pose")
        results.append(check("draw_reaction: unknown pose name falls back safely", True))
    except Exception as e:
        results.append(check(f"draw_reaction: unknown pose name falls back safely (raised {e!r})", False))

    print()
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

