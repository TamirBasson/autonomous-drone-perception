"""Phase 2 validation: confirm cleaned frames exist and are readable."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import load_frames  # noqa: E402


def main() -> int:
    frames = load_frames(REPO_ROOT / "drones_images_input")
    clean_dir = REPO_ROOT / "outputs" / "clean_frames"
    if not clean_dir.is_dir():
        print(f"FAIL: missing {clean_dir}")
        return 1
    missing = [f.name for f in frames if not (clean_dir / f.name).is_file()]
    if missing:
        print(f"FAIL: missing {len(missing)} cleaned frames.")
        return 1
    sample = cv2.imread(str(clean_dir / frames[0].name))
    if sample is None:
        print("FAIL: cannot read cleaned sample image.")
        return 1
    print(f"PASS: validated {len(frames)} cleaned frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
