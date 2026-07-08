"""
Local tests for FaceRecognition — enrollment, caching, FR gating.

Tests:
  1. Legacy compare_faces (no session_id) — backward compatibility
  2. Enrollment via enroll()
  3. Verify after enrollment (first call — full FR)
  4. Verify immediately after (FR skipped — cached similarity)
  5. Verify after recognition_interval elapses (full FR again)
  6. compare_faces with session_id (auto-enroll + cached verify)
  7. Verify with no enrollment — error path

Usage (from AI/ root):
    python -m Models.Face_Recognition_Service.face_recognition_test
"""

import os
import sys
import time

# Ensure AI/ is on sys.path so 'Models.FaceAntiSpoofing' resolves
_AI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _AI_ROOT not in sys.path:
    sys.path.insert(0, _AI_ROOT)

from Models.Face_Recognition_Service.face_recognition import FaceRecognition
import cv2

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    img = cv2.imread(os.path.join(_SERVICE_DIR, name))
    if img is None:
        print(f"ERROR: Could not load {name}")
        sys.exit(1)
    return img


def _print_result(label: str, result: dict):
    print(f"\n--- {label} ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()


def main():
    reference = _load("me.jpg")
    frame = _load("real1.jpg")

    # Use a short recognition_interval (1 s) so FR-gating is testable
    fr = FaceRecognition(recognition_interval=1.0)

    # ------------------------------------------------------------------
    # 1. Legacy compare_faces (no session_id) — backward compat
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: Legacy compare_faces (no session_id)")
    print("=" * 60)
    result = fr.compare_faces(frame, reference)
    _print_result("Legacy compare_faces", result)

    # ------------------------------------------------------------------
    # 2. Enrollment
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 2: Enroll session")
    print("=" * 60)
    enroll_result = fr.enroll("test-session", [reference])
    print(f"  Enrollment result: {enroll_result}")
    assert enroll_result["success"], "Enrollment failed!"

    # ------------------------------------------------------------------
    # 3. Verify (first call — FR runs immediately)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3: Verify after enrollment (first call — full FR)")
    print("=" * 60)
    result = fr.verify("test-session", frame)
    _print_result("Verify (first call)", result)

    # ------------------------------------------------------------------
    # 4. Verify immediately again (FR should be SKIPPED — cached similarity)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 4: Verify immediately (FR skipped — cached similarity)")
    print("=" * 60)
    result2 = fr.verify("test-session", frame)
    _print_result("Verify (cached)", result2)

    # ------------------------------------------------------------------
    # 5. Wait for recognition_interval, then verify (full FR again)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 5: Wait for recognition_interval, then verify (full FR)")
    print("=" * 60)
    print("  Waiting 1.1 seconds ...")
    time.sleep(1.)
    result3 = fr.verify("test-session", frame)
    _print_result("Verify (after interval)", result3)

    # ------------------------------------------------------------------
    # 6. compare_faces with session_id (auto-enroll + cached verify)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 6: compare_faces with session_id (auto-enroll + cache)")
    print("=" * 60)
    result4 = fr.compare_faces(frame, reference, session_id="auto-session")
    _print_result("compare_faces (auto-enroll)", result4)

    # Second call — uses cached enrollment, delegates to verify
    result5 = fr.compare_faces(frame, reference, session_id="auto-session")
    _print_result("compare_faces (cached session)", result5)

    # ------------------------------------------------------------------
    # 7. Verify without enrollment — error
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 7: Verify without enrollment")
    print("=" * 60)
    result6 = fr.verify("nonexistent-session", frame)
    _print_result("Verify (no enrollment)", result6)

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()