# AirCAD - Hands-Free 3D Modeling Using Computer Vision

AirCAD turns a plain webcam into a 3D input device. Pinch to grab a cube, sphere or cylinder in an interactive 3D workspace, move and rotate it with your hand, and scale it by pulling two hands apart - no mouse or keyboard needed for the core interaction.

It grew out of an earlier OpenCV + MediaPipe project, **Hand Gesture Drawing App** (draw in the air with your index finger). That app still works (`python main.py --legacy-draw`) and now shares AirCAD's vision modules.

## Problem statement

3D modeling tools assume a mouse, a keyboard and a flat screen. A webcam sees only a 2D image and no real depth, so "reaching into" 3D space is a genuine mapping problem. AirCAD solves it with a virtual interaction plane, scale-invariant gesture measurements and relative (not absolute) hand motion, and keeps the design open for real depth estimation later.

## Features

- Real 3D workspace (Ursina): perspective camera, grid, X/Y/Z axes, lighting
- Cube, sphere and cylinder entities with create / select / move / rotate / scale / delete
- Pinch-to-grab with hysteresis, debounce and hand-loss grace period
- Two-hand pinch scaling, Move/Rotate modes switched by a three-finger gesture
- Pinch the on-screen buttons to create, delete and reset (hands-free UI)
- Webcam preview, live gesture/mode/selection readout, sensitivity sliders
- Capture + inference on a background thread, so rendering never waits for the camera
- Pure-NumPy gesture/mapping core with unit tests that need no webcam

## Gesture reference

| Gesture | Action |
|---|---|
| Thumb-index pinch on an object | Select and grab it |
| Move pinched hand (Move mode) | Move object in a plane parallel to the screen |
| Move pinched hand / roll wrist (Rotate mode) | Up/down = pitch, left/right = yaw, wrist roll = roll |
| Release pinch | Release object (stays selected) |
| Both hands pinching, distance changes | Uniform scale (ratio of hand distance) |
| Open palm while holding | Release |
| Open palm while idle with a selection | Deselect |
| Three fingers (index, middle, ring; pinky curled) | Toggle Move / Rotate mode (only when nothing is held) |
| Pinch over an on-screen button | Press it (Add Cube/Sphere/Cylinder, Delete, Reset) |

Keyboard/mouse fallbacks (not needed for core interaction): `1`/`2`/`3` add shapes, `Delete` removes the selection, `R` resets, arrow keys orbit the camera, scroll zooms, `F1` shows picking debug dots, `Esc` quits.

## Architecture

```
 Webcam ──► VisionWorker (thread) ──► HandTracker (MediaPipe HandLandmarker, 21 landmarks)
                                         │ latest VisionFrame (hands + preview)
                                         ▼
 HandAssigner ─► GestureRecognizer ×2 ─► InteractionManager (state machine) ─► Commands
 (stable A/B)    (hysteresis+debounce)    ▲ SceneSnapshot                          │
                                          │                                        ▼
                             ObjectController.snapshot()                 ObjectController.apply()
                                          ▲                                        │
                                          └──────────── Scene (Ursina entities) ◄──┘
                                                         + WorkspaceCamera + AirCADUI
```

```
AirCAD/
├── main.py                     CLI entry point
├── app.py                      Ursina app: wires everything together
├── vision/
│   ├── hand_tracker.py         camera, model download, MediaPipe, background worker
│   ├── landmark_processor.py   pure geometry: scale, pinch ratio, finger extension, roll, hand identity
│   └── gesture_recognition.py  static classification + debounced recognizer
├── controls/
│   ├── gesture_mapping.py      coordinate mapping, translation / rotation / scale math
│   ├── interaction_manager.py  Idle / Grabbing / Scaling state machine -> Commands
│   └── object_controller.py    snapshots the scene, applies Commands
├── graphics/
│   ├── scene.py  objects.py  camera.py  ui.py  mesh_data.py
├── utils/        config.py (all tunables), smoothing.py
├── legacy/       gesture_drawing.py (the original app on shared modules)
└── tests/        unittest suites with synthetic landmarks
```

Vision, gesture logic and rendering are separate: `vision/` (except `hand_tracker.py`), `controls/` and `utils/` import neither OpenCV, MediaPipe nor Ursina.

## Technology stack

Python 3.9+, OpenCV (capture, preview), MediaPipe Tasks HandLandmarker (21 landmarks), NumPy (math), Ursina Engine (3D rendering, UI).

## Installation (Windows, PowerShell + VS Code)

```powershell
cd AirCAD                       # the folder that directly contains main.py
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code: **File > Open Folder** on that same folder, then `Ctrl+Shift+P` > **Python: Select Interpreter** > the one in `venv`. If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

On first run the app downloads a small hand-tracking model (~8 MB) into `models/`.

## Run

```powershell
python main.py                 # full app with webcam
python main.py --no-camera     # 3D workspace only (keyboard/mouse)
python main.py --legacy-draw   # original air-drawing app
python main.py --camera-index 1
```

Tests (no webcam or Ursina needed):

```powershell
python -m unittest discover -s tests -t . -v
```

## Computer vision pipeline and math

**1. Detection.** MediaPipe returns 21 landmarks per hand, normalized to the image: x, y in [0, 1] (y down), z relative.

**2. Metric space.** x and y are fractions of different lengths, so distances use `p = (x * aspect, y)` with `aspect = width / height`.

**3. Scale-invariant measurements.** Hand size `s = |p_wrist - p_middleMCP|`. Pinch ratio `r = |p_thumbTip - p_indexTip| / s`. Because of the division, thresholds do not depend on how far the hand is from the camera.

**4. Gestures.** Pinch starts when `r < pinch_on` and ends when `r > pinch_off` (hysteresis prevents flicker). A finger is extended if `|p_wrist - p_tip| > k * |p_wrist - p_pip|` (distance comparison, so it works when the hand is rotated). Poses are mutually exclusive by priority: pinch > open palm > three fingers > neutral. A new gesture must persist `confirm_frames` frames (debounce); one-shot actions are edge-triggered with cooldowns; if the hand vanishes the last gesture is held for `lost_grace_s`, and when it reappears the grab is re-based so the object never teleports.

**5. Virtual interaction plane.** The pinch point (thumb/index midpoint) is mapped from the central part of the frame to `[-1, 1]^2`:
`u = clip(2 * (x - m) / (1 - 2m) - 1)`, `v = clip(1 - 2 * (y - m) / (1 - 2m))`, with margin `m`. The margin means you never have to reach the camera edge.

**6. Smoothing.** Exponential moving average `s_t = a * x_t + (1 - a) * s_(t-1)`.

**7. Relative transforms.** With `d` = (plane position now - at grab time), x scaled by the aspect ratio so horizontal and vertical hand travel feel equal:
- Move: `P = P0 + k_t * (d_x * right + d_y * up)` (camera axes from the viewing camera)
- Rotate: `pitch = pitch0 + k_r * d_y`, `yaw = yaw0 - k_r * d_x`, `roll = roll0 + g * (hand roll change)`, where hand roll is the angle of wrist -> middle MCP
- Scale: `S = clip(S0 * (D / D0) ^ gain)` with `D` the distance between the two pinch points

**8. Picking.** Each object is projected to screen space with the renderer's own lens; the cursor selects the nearest object whose on-screen disc contains it. UI buttons use the same cursor.

**Depth.** A standard webcam gives no metric depth, so motion happens in a plane parallel to the screen. `depth_gain` (default 0 = off) enables an experimental cue: the hand's apparent size relative to grab time. It is relative and noisy; a depth-estimation model could replace it behind `CoordinateMapper.depth_offset`.

## Screenshots

_Add screenshots or a GIF here after running the app (for example `docs/workspace.png`, `docs/two-hand-scale.gif`)._

## Verification status

- **Run and passing:** 67 unit tests (gesture classification, hysteresis, debounce, hand-loss handling, hand identity, normalization, translation/rotation/scale math, smoothing, scripted state-machine scenarios, controller, cylinder mesh) using synthetic landmarks.
- **Import-checked only:** `vision/hand_tracker.py` and the legacy drawing app import cleanly with the installed MediaPipe, but were not run against a camera.
- **Not run:** the Ursina application (`app.py`, `graphics/*`) and live webcam control were written against Ursina's API but not executed during development. Expect to tune small things on first launch: the F1 debug dots show whether picking lines up with the objects; `preview_flip_y` in `utils/config.py` fixes an upside-down preview; sensitivity sliders adjust feel.

## Known limitations

- 2D camera, no real depth: movement is in a plane parallel to the screen.
- Euler-angle rotation: combined pitch/yaw/roll is not a true trackball and can feel odd after large rotations.
- A tightly clenched fist can be read as a pinch on some hands; raise or lower the Pinch threshold slider.
- Hand tracking quality depends on lighting and a plain background; fast motion blurs landmarks.
- Two hands overlapping can swap identities.
- Single viewing camera angle for picking; objects overlapping on screen resolve to the nearest centre.

## Future improvements

- Monocular depth estimation for true 3D translation
- Quaternion trackball rotation, per-axis constraint gestures
- Undo/redo, save/load scenes, export to OBJ/STL
- More primitives and non-uniform scaling
- Gesture calibration wizard per user
- Rendering-independent recording/replay of landmark streams for regression tests
