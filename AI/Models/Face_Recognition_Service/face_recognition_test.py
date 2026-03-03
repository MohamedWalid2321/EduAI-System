"""
Quick local test for the hybrid FaceRecognition module.

Usage:
    cd AI/Models/Face_Recognition_Service
    python face_recognition_test.py
"""

from face_recognition import FaceRecognition
import cv2

# Initialize (loads RetinaFace detector + ArcFace ONNX recogniser)
fr = FaceRecognition()

# Load images
reference = cv2.imread("ronaldo1.jpg")   # Authorised person's ID photo
frame = cv2.imread("ronaldo2.jpg")       # Current webcam frame to verify

if reference is None:
    print("ERROR: Could not load ronaldo1.jpg")
    exit(1)
if frame is None:
    print("ERROR: Could not load ronaldo2.jpg")
    exit(1)

# Compare (returns the standard proctoring module result dict)
result = fr.compare_faces(frame, reference)

print("\n--- Face Recognition Result ---")
for key, value in result.items():
    print(f"  {key}: {value}")
print()

if result["flag"]:
    print("CHEATING DETECTED")
else:
    print("IDENTITY VERIFIED — No cheating")