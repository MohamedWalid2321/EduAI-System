# ObjectDetectionModel.py
from ultralytics import YOLO

# Load YOLO model
def load_model(model_path):
    print("Loading YOLO model...")
    model = YOLO(model_path)
    return model

# List of objects considered cheating
cheating_classes = ['Earphone', 'Mobile_phone', 'headset',
                    'smart_watch', 'sunglasses', 'cap']

# Detect objects in a frame
def detect_objects(model, frame, conf=0.5):
    """
    Returns:
        annotated_frame: frame with bounding boxes
        cheating_detected: bool
        detected_objects: list of detected class names
    """
    results = model(frame, verbose=False, conf=conf)
    cheating_detected = False
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            detected_objects.append(cls_name)
            if cls_name in cheating_classes:
                cheating_detected = True

    annotated_frame = results[0].plot()
    return annotated_frame, cheating_detected, detected_objects
