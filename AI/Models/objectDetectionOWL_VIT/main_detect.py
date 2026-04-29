import datetime
from PIL import Image
from transformers import pipeline

print("Loading OWL-ViT model into memory... (This will take a moment)")
detector = pipeline(
    task="zero-shot-object-detection", 
    model="google/owlvit-base-patch32",
    device=0
)

CHEATING_CLASSES = [
    "Mobile phone", "Earphone", "headset", "smart watch", "sunglasses", "cap"
]

def analyze_proctoring_frame(image: Image.Image) -> dict:
    """
    Analyzes a single frame for restricted objects and returns a formatted dict.
    """
    predictions = detector(
        image,
        candidate_labels=CHEATING_CLASSES,
    )
    max_probability = 0.0
    detected_items = []
    for prediction in predictions:
        score = prediction["score"]
        if score > 0.1: 
            detected_items.append(prediction["label"])
            if score > max_probability:
                max_probability = score

    if detected_items:
        unique_items = list(set(detected_items))
        evidence_str = f"Detected: {', '.join(unique_items)}"
    else:
        evidence_str = "No restricted items detected."
    response = {
        "id": 2, 
        "timestamp": datetime.datetime.now().isoformat(),
        "probability": round(max_probability, 4) if max_probability > 0 else 0.0,
        "evidence": evidence_str            
    }
    
    return response