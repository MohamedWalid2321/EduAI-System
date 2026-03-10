import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from Models.objectDetectionOWL_VIT.main_detect import analyze_proctoring_frame

router = APIRouter(prefix="/analysis", tags=["detect_objects"])

@router.post("/detect_objects")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint to receive a frame, run OWL-ViT detection, and return proctoring evidence.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image format (jpeg, png, etc.).")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = analyze_proctoring_frame(image)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")