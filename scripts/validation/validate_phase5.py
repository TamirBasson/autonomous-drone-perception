"""Phase 5 validation: RANSAC fundamental matrix sanity check."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import (  # noqa: E402
    estimate_fundamental,
    extract_features_for_frames,
    load_frames,
    match_pair,
)


def main() -> int:
    frames = load_frames(REPO_ROOT / "drones_images_input")
    feature_sets = extract_features_for_frames(
        frames,
        method="superpoint",
        use_mask=True,
        source_dir=REPO_ROOT / "outputs" / "clean_frames",
    )
    mr = match_pair(feature_sets[0], feature_sets[1], idx_a=0, idx_b=1, pipeline="superpoint")
    rr = estimate_fundamental(mr)
    if not rr.f_estimated or rr.F is None:
        print("FAIL: fundamental matrix estimation failed for pair (0,1).")
        return 1
    print(f"PASS: F estimated with {rr.num_inliers} inliers for pair (0,1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
