"""Transfer result model and visualization helpers for deep-only flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


DEFAULT_PATCH_SIZE = 21
DEFAULT_STEP = 1.0


@dataclass
class TransferResult:
    """Outcome of a single point-transfer query."""

    source_pixel: Tuple[float, float]
    epipolar_line: np.ndarray
    samples: np.ndarray
    scores: np.ndarray
    predicted_pixel: Optional[Tuple[float, float]] = None
    score: float = float("nan")
    patch_size: int = DEFAULT_PATCH_SIZE
    step: float = DEFAULT_STEP
    source_patch_valid: bool = False
    success: bool = False
    note: str = ""

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0]) if self.samples.size else 0

    @property
    def num_scored(self) -> int:
        if self.scores.size == 0:
            return 0
        return int(np.isfinite(self.scores).sum())


def compute_epipolar_line(
    source_pixel: Tuple[float, float],
    F: np.ndarray,
    source_is_a: bool = True,
) -> np.ndarray:
    """Compute the target-image epipolar line for a source pixel."""
    pt = np.asarray(source_pixel, dtype=np.float32).reshape(1, 1, 2)
    which = 1 if source_is_a else 2
    line = cv2.computeCorrespondEpilines(pt, which, F)
    if line is None:
        raise ValueError("cv2.computeCorrespondEpilines returned None.")
    return line.reshape(3).astype(np.float64)


def _line_segment_in_image(
    line_abc: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Clip the infinite line to image borders."""
    h, w = image_shape
    a, b, c = map(float, line_abc)
    eps = 1e-12
    pts = []

    if abs(b) > eps:
        y = -(a * 0 + c) / b
        if 0 <= y <= h - 1:
            pts.append((0.0, y))
        y = -(a * (w - 1) + c) / b
        if 0 <= y <= h - 1:
            pts.append((w - 1.0, y))
    if abs(a) > eps:
        x = -(b * 0 + c) / a
        if 0 <= x <= w - 1:
            pts.append((x, 0.0))
        x = -(b * (h - 1) + c) / a
        if 0 <= x <= w - 1:
            pts.append((x, h - 1.0))

    uniq = []
    for p in pts:
        if not any((abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6) for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    p1, p2 = uniq[0], uniq[1]
    return (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))


def draw_transfer(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    result: TransferResult,
    ground_truth: Optional[Tuple[float, float]] = None,
    draw_samples: bool = True,
) -> np.ndarray:
    """Draw source point, epipolar line, predicted point and optional GT."""
    vis_src = image_src.copy()
    vis_dst = image_dst.copy()

    u, v = map(int, map(round, result.source_pixel))
    cv2.drawMarker(vis_src, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    seg = _line_segment_in_image(result.epipolar_line, image_dst.shape[:2])
    if seg is not None:
        cv2.line(vis_dst, seg[0], seg[1], (0, 255, 255), 2, cv2.LINE_AA)

    if draw_samples and result.samples.size:
        for x, y in result.samples.astype(np.int32):
            cv2.circle(vis_dst, (int(x), int(y)), 1, (255, 255, 0), -1, cv2.LINE_AA)

    if result.predicted_pixel is not None:
        pu, pv = map(int, map(round, result.predicted_pixel))
        cv2.drawMarker(vis_dst, (pu, pv), (0, 255, 0), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    if ground_truth is not None:
        gu, gv = map(int, map(round, ground_truth))
        cv2.drawMarker(vis_dst, (gu, gv), (255, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    left = vis_src
    right = vis_dst
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        pad = np.zeros((h - left.shape[0], left.shape[1], 3), dtype=left.dtype)
        left = np.vstack([left, pad])
    if right.shape[0] != h:
        pad = np.zeros((h - right.shape[0], right.shape[1], 3), dtype=right.dtype)
        right = np.vstack([right, pad])
    canvas = np.hstack([left, right])

    status = "OK" if result.success else "FAIL"
    score_txt = f"{result.score:.3f}" if np.isfinite(result.score) else "nan"
    text = (
        f"{status} | score={score_txt} | samples={result.num_samples} "
        f"scored={result.num_scored} | {result.note}"
    )
    cv2.putText(
        canvas,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas
