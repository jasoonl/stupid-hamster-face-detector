# stupid-hamster-face-detector

make a face -- shows one of your own reference photos ("hamster" faces)
when your live expression matches it, with a live camera feed + neon-green
readout on the left and the matched photo on the right.

HOW RECOGNITION WORKS NOW (this is a full rewrite of the old approach):

The previous version compared raw pixels of a mouth crop against each
reference photo. Tested against real webcam footage, that failed badly
(~25% -- it literally could not tell a neutral face from a shocked one),
for two reasons: (1) it located the mouth as a fixed fraction of the
OpenCV Haar face box, whose height swings wildly frame to frame, so the
crop landed on the nose or chin; (2) even perfectly aligned, grayscale
mouth crops don't discriminate expressions -- they all correlate.

This version instead:
  - tracks 68 facial landmarks with OpenCV's LBF facemark (pure CPU, so
    it CANNOT trigger the MediaPipe Metal-GPU crash that aborted the old
    app on macOS -- MediaPipe is gone entirely here),
  - turns them (+ a mouth-color read + skin coverage around the face for
    the hand gestures) into a small feature vector,
  - classifies that vector with a RandomForest trained on real labeled
    frames. On the reference footage this separated all five expressions
    (neutral / shock / tongue / huh / shush) at ~94%.

See expression_engine.py for the recognition brain. This file is the
camera loop, the UI, and the in-app "teach it" data-collection mode.

TEACH IT YOUR FACE (improves accuracy):
  While it's running, hold an expression and press the number key for it:
      1 neutral   2 shock   3 tongue   4 huh   5 shush
  Each press records the current frame as a labeled example and instantly
  retrains, so it gets better at YOUR face, lighting and camera the more
  you feed it. Samples are saved (features_user.npz) and reused next run.

CONTROLS:
  1..5  record the current frame as neutral/shock/tongue/huh/shush
  q     quit

SETUP:
  pip3 uninstall -y opencv-python                 # if you have the plain one
  pip3 install "opencv-contrib-python<5" numpy scikit-learn
  python3 app.py            # (add a number, e.g. `python3 app.py 1`, for a second camera)

Put your photos in an "images" folder next to this script, named for the
pose: shock.jpg, tongue.jpg (or toungue.jpg), huh.jpg, shush.jpg,
neutral.jpg. The face-landmark model downloads itself once on first run.
