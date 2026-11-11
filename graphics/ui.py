"""On-screen UI built with Ursina's own widgets (buttons, texts, sliders, webcam preview)."""
from __future__ import annotations

from typing import Callable, Dict, List

from PIL import Image
from ursina import Button, Entity, Slider, Text, Texture, camera, color, window

from controls.interaction_manager import UiTarget
from utils.config import AppConfig

BUTTONS = (("cube", "Add Cube"), ("sphere", "Add Sphere"), ("cylinder", "Add Cylinder"),
           ("delete", "Delete"), ("reset", "Reset Scene"))


class AirCADUI:
    def __init__(self, config: AppConfig, on_action: Dict[str, Callable[[], None]],
                 camera_available: bool) -> None:
        self.config = config
        self._flip = config.camera.preview_flip_y
        aspect = window.aspect_ratio
        left, right = -0.5 * aspect, 0.5 * aspect

        # --- buttons (pinch them with your hand, or click) ---
        self.buttons: Dict[str, Button] = {}
        for i, (name, label) in enumerate(BUTTONS):
            btn = Button(text=label, parent=camera.ui, position=(left + 0.12, 0.40 - i * 0.075),
                         scale=(0.19, 0.055), color=color.dark_gray if name not in ("delete", "reset") else color.brown)
            btn.on_click = on_action[name]
            self.buttons[name] = btn

        # --- webcam preview ---
        cx, cy = right - 0.19, 0.5 - 0.15
        Entity(parent=camera.ui, model="quad", color=color.black, scale=(0.34, 0.26), position=(cx, cy, 0.01))
        self.preview = Entity(parent=camera.ui, model="quad", color=color.white,
                              scale=(0.32, 0.24), position=(cx, cy, 0))
        self.preview_note = Text("" if camera_available else "No camera", parent=camera.ui,
                                 position=(cx, cy), origin=(0, 0), color=color.light_gray)

        # --- status texts ---
        tx = right - 0.35
        self.text_mode = Text("", parent=camera.ui, position=(tx, 0.31), origin=(-0.5, 0.5))
        self.text_gesture = Text("", parent=camera.ui, position=(tx, 0.27), origin=(-0.5, 0.5))
        self.text_selection = Text("", parent=camera.ui, position=(tx, 0.22), origin=(-0.5, 0.5))
        self.text_error = Text("", parent=camera.ui, position=(left + 0.02, -0.40), origin=(-0.5, -0.5),
                               color=color.salmon)
        Text("Pinch: grab | 2 hands pinch: scale | 3 fingers: switch Move/Rotate | open palm: release/deselect",
             parent=camera.ui, position=(0, -0.47), origin=(0, -0.5), scale=0.8, color=color.light_gray)

        # --- sensitivity sliders ---
        self.sliders: List[Slider] = []
        sx, sy = left + 0.05, -0.12
        m, g = config.mapping, config.gesture
        self._add_slider("Move speed", 1.0, 15.0, m.translate_sensitivity, sx, sy,
                         lambda v: setattr(m, "translate_sensitivity", v))
        self._add_slider("Rotate speed", 45.0, 540.0, m.rotate_sensitivity_deg, sx, sy - 0.07,
                         lambda v: setattr(m, "rotate_sensitivity_deg", v))
        self._add_slider("Smoothing (low=smooth)", 0.1, 1.0, m.smoothing_alpha, sx, sy - 0.14,
                         lambda v: setattr(m, "smoothing_alpha", max(0.05, v)))
        self._add_slider("Pinch threshold", 0.15, 0.5, g.pinch_on, sx, sy - 0.21, self._set_pinch)

    # ----------------------------------------------------------------- helpers
    def _set_pinch(self, value: float) -> None:
        self.config.gesture.pinch_on = value
        self.config.gesture.pinch_off = value + 0.15

    def _add_slider(self, text: str, lo: float, hi: float, default: float, x: float, y: float,
                    on_change: Callable[[float], None]) -> None:
        slider = Slider(lo, hi, default=default, text=text, dynamic=True, position=(x, y), parent=camera.ui)
        slider.on_value_changed = lambda s=slider: on_change(s.value)
        self.sliders.append(slider)

    def ui_targets(self) -> List[UiTarget]:
        return [UiTarget(name, (b.x, b.y), (b.scale_x / 2, b.scale_y / 2)) for name, b in self.buttons.items()]

    # ------------------------------------------------------------------ update
    def set_preview(self, rgb) -> None:
        image = Image.fromarray(rgb).convert("RGBA")
        if self._flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.preview.texture = Texture(image)

    def set_status(self, mode: str, state: str, gesture: str, selection: str, error: str = "") -> None:
        self.text_mode.text = f"Mode: {mode}   State: {state}"
        self.text_gesture.text = f"Gesture: {gesture}"
        self.text_selection.text = selection
        self.text_error.text = error
