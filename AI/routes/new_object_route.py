import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

# Import the Modal class from your main file
from main import Proctoring

router = APIRouter(prefix="/analysis", tags=["detect_objects"])

@router.post("/detect_objects")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint to receive a frame, run OWL-ViT detection via Modal GPU, and return proctoring evidence.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image format (jpeg, png, etc.).")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Instantiate the Modal class and call the method natively
        proctor = Proctoring()
        result = proctor.analyze_proctoring_frame.local(image)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")