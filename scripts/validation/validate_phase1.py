"""Phase 1 validation: confirm input frames are readable."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import load_frames  # noqa: E402


def main() -> int:
    frames = load_frames(REPO_ROOT / "drones_images_input")
    if not frames:
        print("FAIL: no input frames found.")
        return 1
    _ = frames[0].load_image()
    print(f"PASS: loaded {len(frames)} input frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
