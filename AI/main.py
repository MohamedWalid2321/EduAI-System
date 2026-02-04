import sys
import os
import cv2
import msvcrt  # For keyboard input on Windows

current_dir = os.path.dirname(os.path.abspath(__file__))
server_path = os.path.join(current_dir, 'Models', 'EyeGazeDetection', 'src', 'Server')
sys.path.append(server_path)

from Models.objectDetectionYolo.objectDetection import yoloDetect
import Models.EyeGazeDetection.src.Server.localMain as GazeMain
from Models.Face_Recognition_Service import FaceRecognition

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Press 'q' to quit.")
    
    fr = FaceRecognition()
    reference = cv2.imread("ronaldo1.jpg")
    
    if reference is None:
        print("Error: Could not load reference image 'ronaldo1.jpg'. Check if file exists.")
        cap.release()
        return
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # Validate frame is not empty
            if frame is None or frame.size == 0:
                print("Empty frame, skipping...")
                continue
                
            results1 = yoloDetect(frame)
            results2 = GazeMain.process_gaze_frame(frame, False)

            result3 = fr.compare_faces(frame, reference)

            # print(f"YOLO: {results1} \n GAZE: {results2}")
            print(results1)
            print(results2)
            print(result3)
            # cv2.imshow('Main System', frame)

            # Check for 'q' key press without blocking
            if msvcrt.kbhit():
                if msvcrt.getch().decode('utf-8', errors='ignore').lower() == 'q':
                    print("Quitting...")
                    break
        except cv2.error as e:
            print(f"OpenCV error (skipping frame): {e}")
            continue
        except Exception as e:
            print(f"Error in loop: {e}")
            import traceback
            traceback.print_exc()
            break

    cap.release()

if __name__ == "__main__":
    main()