"""Phase 3 validation: SuperPoint extraction on cleaned frames."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import extract_features_for_frames, load_frames  # noqa: E402


def main() -> int:
    frames = load_frames(REPO_ROOT / "drones_images_input")
    feature_sets = extract_features_for_frames(
        frames,
        method="superpoint",
        use_mask=True,
        source_dir=REPO_ROOT / "outputs" / "clean_frames",
    )
    counts = [fs.num_keypoints for fs in feature_sets]
    if min(counts) <= 0:
        print("FAIL: one or more frames produced zero keypoints.")
        return 1
    print(f"PASS: extracted features for {len(feature_sets)} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
