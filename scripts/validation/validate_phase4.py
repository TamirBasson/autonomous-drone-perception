"""Phase 4 validation: LightGlue matching sanity check."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import extract_features_for_frames, load_frames, match_pair  # noqa: E402


def main() -> int:
    frames = load_frames(REPO_ROOT / "drones_images_input")
    feature_sets = extract_features_for_frames(
        frames,
        method="superpoint",
        use_mask=True,
        source_dir=REPO_ROOT / "outputs" / "clean_frames",
    )
    res = match_pair(feature_sets[0], feature_sets[1], idx_a=0, idx_b=1, pipeline="superpoint")
    if res.num_tentative <= 0:
        print("FAIL: no tentative matches for pair (0,1).")
        return 1
    print(f"PASS: pair (0,1) has {res.num_tentative} tentative matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
