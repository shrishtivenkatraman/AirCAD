"""Central, easily-tweakable configuration for AirCAD.

Every tunable number lives here so that gesture feel, mapping sensitivity and
camera settings can be adjusted without touching algorithm code. The dataclasses
are mutable on purpose: the UI sliders edit them at runtime and the recognizers
and mappers read them every frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True            # selfie view: moving your hand right moves the cursor right
    preview_width: int = 256       # width of the webcam thumbnail shown in the UI
    preview_fps: float = 15.0      # how often the UI texture is refreshed
    preview_flip_y: bool = False   # set True if the preview appears upside down


@dataclass
class GestureConfig:
    # Pinch = thumb-tip/index-tip distance divided by hand size (wrist -> middle MCP).
    pinch_on: float = 0.30         # ratio below which a pinch STARTS
    pinch_off: float = 0.45        # ratio above which a pinch ENDS (hysteresis)
    extend_ratio: float = 1.05     # finger counts as extended if tip is this much farther from wrist than the PIP joint
    confirm_frames: int = 3        # consecutive frames a new gesture must persist (debounce)
    lost_grace_s: float = 0.35     # keep the last gesture alive this long when the hand vanishes
    mode_switch_cooldown_s: float = 1.0
    deselect_cooldown_s: float = 0.8
    click_cooldown_s: float = 0.6

    def validate(self) -> None:
        if not 0.0 < self.pinch_on < self.pinch_off:
            raise ValueError("pinch_on must be > 0 and smaller than pinch_off")
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")


@dataclass
class MappingConfig:
    active_margin: float = 0.12        # fraction of the frame ignored at each edge (virtual interaction plane)
    translate_sensitivity: float = 5.0 # world units per plane unit of hand travel
    rotate_sensitivity_deg: float = 180.0  # degrees per plane unit of hand travel
    roll_gain: float = 1.0             # object roll per degree of hand roll
    depth_gain: float = 0.0            # experimental hand-size depth cue; 0 disables it
    scale_gain: float = 1.0            # exponent applied to the two-hand distance ratio
    min_scale: float = 0.25
    max_scale: float = 4.0
    smoothing_alpha: float = 0.35      # EMA factor: 1 = no smoothing, small = very smooth
    pick_radius: float = 0.10          # minimum selection radius, in UI units (screen height = 1)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    window_title: str = "AirCAD - Hands-Free 3D Modeling"

    def __post_init__(self) -> None:
        self.gesture.validate()
