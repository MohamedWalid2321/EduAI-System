"""
Face Recognition Service using DeepFace
This module provides face recognition capabilities for exam proctoring.
"""
"""
┌─────────────────────────────────────────────────────────┐
│           FACE RECOGNITION SYSTEM                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────────────┐                                   │
│   │  Haar Cascade   │ ──► Draw bounding box (UI)        │
│   │  (OpenCV)       │     Fast, every frame             │
│   └─────────────────┘                                   │
│                                                          │
│   ┌─────────────────┐                                   │
│   │      SSD        │ ──► Detect face for recognition   │
│   │  (DeepFace)     │     Accurate, background thread   │
│   └────────┬────────┘                                   │
│            │                                             │
│            ▼                                             │
│   ┌─────────────────┐                                   │
│   │    VGG-Face     │ ──► Recognize identity            │
│   │  (DeepFace)     │     Compare with reference        │
│   └─────────────────┘                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
"""
import cv2
import os
import numpy as np
from deepface import DeepFace


class FaceRecognitionService:
    def __init__(self, reference_images_path="reference_images", model_name="VGG-Face", detector_backend="retinaface"):
        """
        Initialize the Face Recognition Service.
        
        Args:
            reference_images_path: Path to directory containing reference images
            model_name: DeepFace model to use (Facenet is faster, VGG-Face is more accurate)
            detector_backend: Face detector (retinaface is most accurate, opencv is fastest)
        """
        self.reference_images_path = reference_images_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.reference_embeddings = {}
        
        # Create reference images directory if it doesn't exist
        if not os.path.exists(reference_images_path):
            os.makedirs(reference_images_path)
    
    def register_face(self, image_path, person_id):
        """
        Register a new face with a person ID.
        
        Args:
            image_path: Path to the face image
            person_id: Unique identifier for the person
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Verify the image contains a face
            face_objs = DeepFace.extract_faces(img_path=image_path, enforce_detection=True)
            
            if len(face_objs) > 0:
                # Copy image to reference folder
                dest_path = os.path.join(self.reference_images_path, f"{person_id}.jpg")
                img = cv2.imread(image_path)
                cv2.imwrite(dest_path, img)
                print(f"Successfully registered face for: {person_id}")
                return True
            return False
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def verify_face(self, image_path, person_id):
        """
        Verify if the face in the image matches the registered person.
        
        Args:
            image_path: Path to the image to verify
            person_id: ID of the person to verify against
            
        Returns:
            dict: Verification result with 'verified' boolean and 'distance' score
        """
        try:
            reference_path = os.path.join(self.reference_images_path, f"{person_id}.jpg")
            
            if not os.path.exists(reference_path):
                return {"verified": False, "error": "Person not registered"}
            
            result = DeepFace.verify(
                img1_path=image_path,
                img2_path=reference_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            return {
                "verified": result["verified"],
                "distance": result["distance"],
                "threshold": result["threshold"],
                "model": self.model_name
            }
        except Exception as e:
            return {"verified": False, "error": str(e)}
    
    def verify_face_from_frame(self, frame, person_id):
        """
        Verify face from a video frame (numpy array).
        
        Args:
            frame: OpenCV frame (numpy array)
            person_id: ID of the person to verify against
            
        Returns:
            dict: Verification result
        """
        try:
            reference_path = os.path.join(self.reference_images_path, f"{person_id}.jpg")
            
            if not os.path.exists(reference_path):
                return {"verified": False, "error": "Person not registered"}
            
            # First check if a face is detected in the frame
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="ssd",
                    enforce_detection=True
                )
                if not faces or len(faces) == 0:
                    return {"verified": False, "error": "Face could not be detected"}
            except Exception:
                # No face detected
                return {"verified": False, "error": "Face could not be detected"}
            
            result = DeepFace.verify(
                img1_path=frame,
                img2_path=reference_path,
                model_name=self.model_name,
                detector_backend="ssd",  # SSD is good balance of speed and accuracy
                enforce_detection=False,  # Don't fail if face not detected
                anti_spoofing=False
            )
            
            # Use slightly relaxed threshold for real-time verification
            distance = result["distance"]
            threshold = result["threshold"] * 1.2  # 20% more lenient
            verified = distance <= threshold
            
            return {
                "verified": verified,
                "distance": distance,
                "threshold": threshold,
                "model": self.model_name
            }
        except Exception as e:
            return {"verified": False, "error": str(e)}
    
    def find_face(self, image_path):
        """
        Find matching face from the reference database.
        
        Args:
            image_path: Path to the image to search
            
        Returns:
            list: List of matching identities with distances
        """
        try:
            results = DeepFace.find(
                img_path=image_path,
                db_path=self.reference_images_path,
                model_name=self.model_name,
                enforce_detection=True
            )

            # results is a list of DataFrames.
            # results[0] is the first DataFrame in that list, [face_name : distance]
            # which contains all the matches found in the reference database for your query image.
            if len(results) > 0 and len(results[0]) > 0:
                matches = []
                for _, row in results[0].iterrows():
                    # row['identity'] → full path to the reference image (e.g., "reference_images/Alice.jpg").
                    # os.path.basename(...) → extracts just the filename ("Alice.jpg").
                    # .replace('.jpg', '') → removes the .jpg extension.
                    # Result: identity now holds "Alice" (the person_id).

                    identity = os.path.basename(row['identity']).replace('.jpg', '')
                    matches.append({
                        "person_id": identity,
                        "distance": row['distance']
                    })
                return matches
            return []
        except Exception as e:
            print(f"Error finding face: {e}")
            return []
    
    def detect_faces(self, image_path):
        """
        Detect all faces in an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            list: List of detected face regions
        """
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                enforce_detection=False
            )
            
            face_regions = []
            for face in faces:
                face_regions.append({
                    "facial_area": face["facial_area"],
                    "confidence": face["confidence"]
                })
            return face_regions
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def analyze_face(self, image_path):
        """
        Analyze face attributes (age, gender, emotion, race).
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Face analysis results
        """
        try:
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=True
            )
            return analysis
        except Exception as e:
            print(f"Error analyzing face: {e}")
            return None


def real_time_verification(person_id, reference_path="reference_images"):
    """
    Run real-time face verification using webcam.
    
    Args:
        person_id: ID of the person to verify
        reference_path: Path to reference images directory
    """
    service = FaceRecognitionService(reference_images_path=reference_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting real-time face verification...")
    print("Press 'q' to quit")
    
    frame_count = 0
    verification_interval = 30  # Verify every 30 frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Perform verification at intervals
        if frame_count % verification_interval == 0:
            result = service.verify_face_from_frame(frame, person_id)
            
            if result.get("verified"):
                status = f"VERIFIED - {person_id}"
                color = (0, 255, 0)  # Green
            else:
                status = f"NOT VERIFIED - {result.get('error', 'Unknown')}"
                color = (0, 0, 255)  # Red
            
            # Store last result for display
            last_result = (status, color)
        
        # Display status on frame
        if 'last_result' in dir():
            cv2.putText(frame, last_result[0], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_result[1], 2)
        
        cv2.imshow('Face Verification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    service = FaceRecognitionService()
    
    print("Face Recognition Service initialized")
    print("Available methods:")
    print("  - register_face(image_path, person_id)") # Register a new face with ID
    print("  - verify_face(image_path, person_id)") # Verify face from image
    print("  - verify_face_from_frame(frame, person_id)") # Verify face from Real Time webcam frame
    print("  - find_face(image_path)") # Find face in image in reference database
    print("  - detect_faces(image_path)") # Detect all faces in image 
    print("  - analyze_face(image_path)") # Analyze face attributes like age, gender, emotion    
    print("\nFor real-time verification, call: real_time_verification(person_id)")
