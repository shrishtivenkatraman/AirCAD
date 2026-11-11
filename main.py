"""AirCAD entry point.

    python main.py                 # launch the 3D workspace with webcam control
    python main.py --no-camera     # mouse/keyboard only (for testing the 3D side)
    python main.py --legacy-draw   # the original air-drawing app (index finger draws)
"""
from __future__ import annotations

import argparse
import sys

from utils.config import AppConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirCAD - hands-free 3D modeling with computer vision")
    p.add_argument("--camera-index", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--no-camera", action="store_true", help="run without the webcam")
    p.add_argument("--legacy-draw", action="store_true", help="run the original hand gesture drawing app")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = AppConfig()
    config.camera.index = args.camera_index

    if args.legacy_draw:
        from legacy.gesture_drawing import main as draw_main
        draw_main(config.camera)
        return 0

    try:
        import ursina  # noqa: F401
    except ImportError:
        print("Ursina is not installed. Run:  pip install -r requirements.txt")
        return 1

    from app import run
    from vision.hand_tracker import VisionWorker

    worker = None
    if not args.no_camera:
        worker = VisionWorker(config.camera)
        if not worker.start():
            print(f"Warning: vision disabled - {worker.error}")
            worker.stop()  # the error message stays on the worker and is shown in the UI
    run(config, worker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
