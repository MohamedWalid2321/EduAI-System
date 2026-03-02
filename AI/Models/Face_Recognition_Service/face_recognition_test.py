from face_recognition import FaceRecognition
import cv2

# Initialize
fr = FaceRecognition()

# Load images
reference = cv2.imread("ronaldo1.jpg")  # Authorized person's photo
frame = cv2.imread("ronaldo2.jpg")          # Current frame to check

# Compare
result = fr.compare_faces(frame, reference)
print(result)