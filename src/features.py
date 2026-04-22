"""Phase 3 feature extraction (SuperPoint only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .frame_loader import Frame
from .preprocessing import (
    DEFAULT_OVERLAY_REGIONS,
    CALIBRATION_SIZE,
    OverlayRegion,
    build_overlay_mask,
)


SUPPORTED_METHODS = ("superpoint",)


@dataclass
class FeatureSet:
    """Keypoints + descriptors extracted from a single frame."""
    frame_name: str
    method: str
    image_shape: Tuple[int, int]
    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)


def build_detection_mask(
    image: np.ndarray,
    regions: Sequence[OverlayRegion] = DEFAULT_OVERLAY_REGIONS,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
) -> np.ndarray:
    """Return a uint8 mask where 255 = 'allowed for keypoint detection'.

    Pixels inside overlay regions (HUD / telemetry) are set to 0 so the
    detector ignores them, even after inpainting.
    """
    overlay = build_overlay_mask(image, regions=regions, calibration_size=calibration_size)
    return cv2.bitwise_not(overlay)


def extract_features(
    image: np.ndarray,
    method: str = "superpoint",
    mask: Optional[np.ndarray] = None,
    frame_name: str = "",
) -> FeatureSet:
    """Detect keypoints and compute descriptors on a single image.

    Parameters
    ----------
    image : BGR or grayscale uint8 image.
    method : must be "superpoint".
    mask : optional uint8 mask (255 = allowed, 0 = ignored).
    frame_name : used only for bookkeeping / debug prints.

    Dispatches to `src.deep_features.extract_superpoint` (lazy import).
    """
    if method.lower() != "superpoint":
        raise ValueError("Only 'superpoint' is supported in this project version.")
    from .deep_features import extract_superpoint
    return extract_superpoint(image, mask=mask, frame_name=frame_name)


def extract_features_for_frames(
    frames: Iterable[Frame],
    method: str = "superpoint",
    use_mask: bool = True,
    regions: Sequence[OverlayRegion] = DEFAULT_OVERLAY_REGIONS,
    calibration_size: Tuple[int, int] = CALIBRATION_SIZE,
    source_dir: Optional[str | Path] = None,
) -> List[FeatureSet]:
    """Extract features for each frame and return a list of `FeatureSet`.

    Parameters
    ----------
    frames : iterable of `Frame` objects (filename is preserved).
    method : detector name.
    use_mask : if True, ignores overlay regions during detection.
    regions, calibration_size : overlay definitions (passed to the mask builder).
    source_dir : if provided, load the image from `<source_dir>/<frame.name>`
                 instead of the original input path. Use this to run on
                 the cleaned frames produced by Phase 2.
    """
    source_dir = Path(source_dir) if source_dir is not None else None
    results: List[FeatureSet] = []

    for frame in frames:
        if source_dir is not None:
            path = source_dir / frame.name
            image = cv2.imread(str(path))
            if image is None:
                raise IOError(f"Failed to read cleaned frame: {path}")
        else:
            image = frame.load_image()

        mask = None
        if use_mask:
            mask = build_detection_mask(
                image, regions=regions, calibration_size=calibration_size,
            )

        fs = extract_features(image, method=method, mask=mask, frame_name=frame.name)
        results.append(fs)

    return results


def apply_grid_filter(fs: FeatureSet, *args, **kwargs) -> FeatureSet:
    """Deprecated in deep-only mode. Returned unchanged for compatibility."""
    return fs


def draw_keypoints(
    image: np.ndarray,
    feature_set: FeatureSet,
    color: Tuple[int, int, int] = (0, 255, 0),
    rich: bool = True,
) -> np.ndarray:
    """Return a copy of `image` with keypoints drawn.

    If `rich=True` (default) each keypoint is drawn with size and orientation
    (useful to visually assess scale/rotation distribution).
    """
    flags = (
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        if rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    return cv2.drawKeypoints(image, feature_set.keypoints, None, color=color, flags=flags)
