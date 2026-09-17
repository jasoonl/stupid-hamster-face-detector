# stupid-hamster-face-detector

# make a face

A webcam toy that shows one of your own reference photos (your "hamster"
faces) when your live expression matches it. Make a shocked face → your
shock photo pops up. Stick your tongue out → your tongue photo. Etc.

Live camera + a neon-green readout on the left, the matched photo on the
right.

## How it works

Recognition is a small **trained classifier**, not pixel matching:

1. **68 facial landmarks** are tracked every frame with OpenCV's LBF
   facemark (`cv2.face`) — a pure-CPU model, so there's no GPU dependency
   and nothing to crash.
2. Those landmarks (plus a color read of the mouth interior and skin
   coverage in zones around the face, for the hand gestures) become a
   small **feature vector** describing the expression.
3. A **RandomForest** classifies that vector into one of five states:
   `neutral`, `shock`, `tongue`, `huh`, `shush`. It's trained at launch
   from a shipped dataset of feature numbers plus anything you teach it.

On the reference footage it separates all five expressions at ~91%
(honest temporal holdout), versus ~25% for the old pixel-matching
approach it replaced.

## Setup

```bash
# If you have plain opencv-python, remove it first (it shadows cv2.face):
pip3 uninstall -y opencv-python

pip3 install -r requirements.txt
python3 app.py
```

Add a camera index for a second camera, e.g. `python3 app.py 1` (useful
if an iPhone Continuity Camera grabs index 0 on a Mac).

The face-landmark model (`lbfmodel.yaml`, ~54MB) downloads itself once on
first run.

## Controls

| key   | action |
|-------|--------|
| `1`   | record the current frame as **neutral** |
| `2`   | record the current frame as **shock** |
| `3`   | record the current frame as **tongue** |
| `4`   | record the current frame as **huh** |
| `5`   | record the current frame as **shush** |
| `q`   | quit |

## Teach it your face

Hold an expression and press its number key. Each press records that
frame as a labeled example and **instantly retrains**, so accuracy climbs
on your face, lighting, and camera the more you feed it. Your samples
save to `features_user.npz` and are reused every run. To reset your
teaching, delete that file.

## Your photos

Put your reference photos in an `images/` folder next to `app.py`, named
for the pose:

```
images/
├── neutral.jpg
├── shock.jpg
├── tongue.jpg   (or toungue.jpg)
├── huh.jpg
└── shush.jpg
```

`.png`, `.jpg`, `.jpeg`, `.webp` all work; transparent PNGs composite
correctly.

## Files

| file | what it is |
|------|-----------|
| `app.py` | camera loop, UI, the "teach it" keys |
| `expression_engine.py` | the recognition brain (landmarks → features → classifier) |
| `features_bootstrap.npz` | starting training data (feature numbers only, no images) so it works out of the box |
| `test_app.py` | tests (`python3 test_app.py`) |
| `images/` | your reference photos |
| `requirements.txt` | dependencies |

`lbfmodel.yaml` (downloaded) and `features_user.npz` (your recordings)
are git-ignored — they're generated locally, not committed.

## Credits

Facial-landmark model: [LBF 68-point model](https://github.com/kurnianggoro/GSOC2017)
trained for OpenCV's `cv2.face` facemark. Built with OpenCV, NumPy, and
scikit-learn.
