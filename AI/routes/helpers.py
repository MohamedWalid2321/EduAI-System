"""
Shared helpers used across all proctoring route modules.
"""

from __future__ import annotations

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List

from fastapi import HTTPException, status

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SECONDS = 7
FPS = 30
TARGET_FRAMES = TARGET_SECONDS * FPS  # 210

# Shared thread-pool for CPU-bound AI inference
executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path: str) -> List[np.ndarray]:
    """Sample up to TARGET_FRAMES evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    if not cap.isOpened():
        raise ValueError("Failed to open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // TARGET_FRAMES, 1)

    idx = 0
    while cap.isOpened() and len(frames) < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1

    cap.release()

    if not frames:
        raise ValueError("No frames extracted")

    return frames


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------
def aggregate(results: list[dict], module_id: int) -> dict:
    """Combine per-frame results into a single summary dict."""
    probabilities = []
    flags = []
    evidence_set: set[str] = set()

    for r in results:
        prob = r.get("propability") or r.get("probability") or 0.0
        probabilities.append(prob)
        flags.append(bool(r.get("flag", False)))

        ev = r.get("evidence")
        if ev:
            for item in ev.split(";"):
                item = item.strip()
                if item:
                    evidence_set.add(item)

    return {
        "id": module_id,
        "timestamp": datetime.now().isoformat(),
        "flag": any(flags),
        "propability": round(max(probabilities), 4),
        "evidence": ", ".join(sorted(evidence_set))
        if evidence_set
        else "No suspicious activity detected",
    }


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def bytes_to_cv2(raw: bytes, *, label: str = "image") -> np.ndarray:
    """Decode raw bytes to a BGR OpenCV image; raise 422 on failure."""
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode {label} — unsupported or corrupt file.",
        )
    return img
