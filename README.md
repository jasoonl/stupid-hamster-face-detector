# stupid-hamster-face-detector

make a face — a live webcam "doodle mirror".

Same idea as the demo it's based on: a green tracking box + a live metrics
readout on one side, a "make a face" prompt on the other that turns into a
simple line-doodle face mirroring your expression in real time.

Tech stack matches the original: plain Python + OpenCV (cv2), no ML model
file, nothing leaves your machine. Detection is a set of hand-built
heuristics on top of OpenCV's own bundled Haar cascades (face / eye /
smile) rather than a trained facial-landmark model — that needs a model
file this environment has no way to fetch or verify, so it's not something
I'll silently fake. What's here is genuinely tested (see test_facelogic.py)
against a real face photo, not just against clean synthetic shapes.

Run:
    pip install opencv-python numpy
    python app.py
Controls:
    c — calibrate (hold a relaxed, neutral face still for ~1 second)
    q — quit
