"""
Test script for Face Recognition Service using test1.jpg
Shows live webcam panel with face verification results
"""

import cv2
import os
import sys
import threading #run face verification in background to keep the webcam live.
from deepface import DeepFace

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_recognition import FaceRecognitionService


def run_face_verification_panel(person_id="Desha"):
    """
    Run real-time face verification with a visible panel.
    Shows webcam feed with verification status overlay.
    """
    # Initialize service with reference images path
    reference_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reference_images")
    service = FaceRecognitionService(
        reference_images_path=reference_path,
        model_name="VGG-Face",  # More accurate for recognition
        detector_backend="ssd"  # Good balance of speed and accuracy
    )
    
    # Check if reference image exists
    ref_image_path = os.path.join(reference_path, f"{person_id}.jpg")
    if not os.path.exists(ref_image_path):
        print(f"ERROR: Reference image not found at: {ref_image_path}")
        print("Please ensure test1.jpg is in the reference_images folder")
        return
    
    print(f"Reference image found: {ref_image_path}")
    
    # Load reference image for display
    ref_img = cv2.imread(ref_image_path)
    #Resizes to 150x150 for displaying in the panel.
    ref_img_small = cv2.resize(ref_img, (150, 150)) if ref_img is not None else None
    
    # Load OpenCV's fast face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "=" * 50)
    print("FACE RECOGNITION VERIFICATION PANEL")
    print("=" * 50)
    print(f"Verifying against: {person_id}")
    print("Press 'q' to quit")
    print("=" * 50 + "\n")
    
    # Shared state for background verification
    verification_result = {"status": "Initializing...", "verified": False, "color": (255, 255, 255), "distance": 0, "threshold": 0}
    verification_lock = threading.Lock()
    verification_running = False # Flag to prevent overlapping verifications
    
    def verify_in_background(frame_to_verify):
        nonlocal verification_running # Use the variable verification_running from the nearest enclosing function, not create a new one
        try:
            result = service.verify_face_from_frame(frame_to_verify, person_id)
            with verification_lock:
                distance = result.get("distance", 0)
                threshold = result.get("threshold", 0)
                verification_result["distance"] = distance
                verification_result["threshold"] = threshold
                
                if result.get("verified"):
                    verification_result["verified"] = True
                    verification_result["status"] = "VERIFIED"
                    verification_result["color"] = (0, 255, 0)
                    print(f"[OK] VERIFIED - Distance: {distance:.4f} (Threshold: {threshold:.4f})")
                else:
                    verification_result["verified"] = False
                    error = result.get("error", "Face mismatch")
                    if "Face could not be detected" in str(error) or "could not be detected" in str(error).lower():
                        verification_result["status"] = "NO FACE DETECTED"
                        verification_result["color"] = (0, 165, 255)
                        print(f"[X] NO FACE DETECTED")
                    else:
                        verification_result["status"] = "NOT MATCHED"
                        verification_result["color"] = (0, 0, 255)
                        print(f"[X] NOT MATCHED - Distance: {distance:.4f} (Threshold: {threshold:.4f})")
        except Exception as e:
            with verification_lock:
                verification_result["status"] = "Error"
                verification_result["color"] = (0, 0, 255)
        finally: # Ensure the flag is reset and it excutes no matter what happens in try block
            verification_running = False
    
    frame_count = 0
    verification_interval = 60  # Start verification every 60 frames (~2 sec)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Fast face detection using Haar Cascade (runs every frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        # Draw face boxes
        with verification_lock:
            box_color = verification_result["color"]
            is_verified = verification_result["verified"]
        
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 3)
            label = "MATCH" if is_verified else "DETECTING..."
            cv2.rectangle(display_frame, (x, y - 30), (x + len(label) * 12, y), box_color, -1)
            cv2.putText(display_frame, label, (x + 5, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Start background verification (non-blocking)
        if frame_count % verification_interval == 0 and not verification_running:
            verification_running = True
            thread = threading.Thread(target=verify_in_background, args=(frame.copy(),))
            thread.daemon = True
            thread.start()
        
        # Get current verification state
        with verification_lock:
            current_status = verification_result["status"]
            current_color = verification_result["color"]
            current_verified = verification_result["verified"]
            current_distance = verification_result["distance"]
            current_threshold = verification_result["threshold"]
        
        # Draw status banner at top
        cv2.rectangle(display_frame, (0, 0), (640, 80), (50, 50, 50), -1)
        
        # Draw verification status
        cv2.putText(display_frame, current_status, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, current_color, 2)
        
        # Draw distance info if available
        if current_distance > 0:
            info_text = f"Distance: {current_distance:.4f} / Threshold: {current_threshold:.4f}"
            cv2.putText(display_frame, info_text, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw reference image in corner
        if ref_img_small is not None:
            # Position in top-right corner
            x_offset = display_frame.shape[1] - 160
            y_offset = 90
            
            # Add border based on verification status
            border_color = current_color
            cv2.rectangle(display_frame, (x_offset - 5, y_offset - 5), 
                         (x_offset + 155, y_offset + 155), border_color, 3)
            
            # Place reference image
            display_frame[y_offset:y_offset + 150, x_offset:x_offset + 150] = ref_img_small
            
            # Label
            cv2.putText(display_frame, "Reference", (x_offset, y_offset + 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw verification indicator circle
        circle_center = (50, 120)
        if current_verified:
            cv2.circle(display_frame, circle_center, 20, (0, 255, 0), -1) #parametes: image, center, radius, color, thickness
            cv2.putText(display_frame, "OK", (38, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) #parameters: image, text, org, font, fontScale, color, thickness
        else:
            cv2.circle(display_frame, circle_center, 20, (0, 0, 255), -1)
            cv2.putText(display_frame, "X", (43, 127),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions at bottom
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Show the panel
        cv2.imshow('Face Recognition - Verification Panel', display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'): # how it works : waitKey(1) returns a 32-bit integer corresponding to the key pressed. The last 8 bits represent the ASCII value of the key.
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nVerification panel closed.")


if __name__ == "__main__":
    print("Starting Face Recognition Test with test1.jpg...")
    run_face_verification_panel("test1")
