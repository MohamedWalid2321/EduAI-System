import cv2
import datetime
from ultralytics import YOLO
import os

# Initialize the model
model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt"))
CHEATING_CLASSES = [
    "Mobile_phone", 
    "Earphone", 
    "headset", 
    "smart_watch", 
    "cap" 
]

def yoloDetect(frame):
    results = model(frame, conf=0.4, verbose=False)
    detected_evidence = []
    max_probability = 0.0
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0]) # Class ID
            conf = float(box.conf[0]) # Confidence
            class_name = result.names[cls_id] # Class name 
            
            if class_name in CHEATING_CLASSES:
                detected_evidence.append(class_name)
                # Update max probability if current confidence is higher
                if conf > max_probability:
                    max_probability = conf
                    
    unique_evidence = list(set(detected_evidence))
    evidence_str = ", ".join(unique_evidence) if unique_evidence else "None"
    
    response = {
        "id": 2, 
        "timestamp": datetime.datetime.now().isoformat(),
        "probability": round(max_probability, 4),  # confidence   
        "evidence": evidence_str            
    }
    
    return response

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()
        
    print("Press 'q' to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        data = yoloDetect(frame)
        print(data)
        
        # cv2.imshow('Webcam Feed', frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()