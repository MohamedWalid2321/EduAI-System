import sys
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
server_path = os.path.join(current_dir, 'models', 'EyeGazeDetection', 'src', 'Server')
sys.path.append(server_path)

from models.objectDetectionYolo.objectDetection import yoloDetect
import models.EyeGazeDetection.src.Server.localMain as GazeMain

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        results1 = yoloDetect(frame)
        results2 = GazeMain.process_gaze_frame(frame, False)
        
        # print(f"YOLO: {results1} \n GAZE: {results2}")
        print(results2)
        # cv2.imshow('Main System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()