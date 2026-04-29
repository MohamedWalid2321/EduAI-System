"""
EduAI Proctoring API — FastAPI + Modal entrypoint.

Run locally:
    conda activate eye_gaze
    uvicorn main:create_app --factory --reload --host 0.0.0.0 --port 8000

Deploy to Modal:
    conda activate eye_gaze
    python -m modal deploy main.py

Dev-serve on Modal (temporary URL, hot-reload):
    conda activate eye_gaze
    python -m modal serve main.py
"""
import os
import sys
import logging

import modal

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


# ---------------------------------------------------------------------------
# Factory: builds the FastAPI app
# ---------------------------------------------------------------------------
def create_app():
    """Build and return the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    )

    application = FastAPI(
        title="EduAI Proctoring API",
        description="AI proctoring — Object detection via OWL-ViT.",
        version="1.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from routes import new_object_route
    application.include_router(new_object_route)

    @application.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok"}

    return application


# ---------------------------------------------------------------------------
# Modal deployment
# ---------------------------------------------------------------------------
app = modal.App("eduai-proctoring")
_modal_gpu = os.getenv("MODAL_GPU", "L4")

modal_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        _current_dir,
        remote_path="/root/app",
        ignore=[
            "__pycache__",
            ".git",
            ".venv",
        ],
    )
)


@app.cls(
    image=modal_image,
    gpu=_modal_gpu,
    scaledown_window=600,
)
class Proctoring:
    """Modal class that keeps the OWL-ViT model warm in memory between requests."""

    @modal.enter()
    def preload(self):
        import sys as _sys
        import logging
        from transformers import pipeline

        if "/root/app" not in _sys.path:
            _sys.path.insert(0, "/root/app")
        
        log = logging.getLogger("preload")
        logging.basicConfig(level=logging.INFO)

        log.info("⏳ Loading OWL-ViT model onto GPU...")
        
        # Initialize it as an instance variable (self.detector) so it stays warm
        self.detector = pipeline(
            task="zero-shot-object-detection", 
            model="google/owlvit-base-patch32",
            device=0,
            framework="pt"  # Forces PyTorch, bypassing the TensorFlow warnings
        )
        
        self.CHEATING_CLASSES = [
            "Mobile phone", "Earphone", "headset", "smart watch", "sunglasses", "cap"
        ]
        log.info("✅ OWL-ViT loaded")

    @modal.method()
    def analyze_proctoring_frame(self, image):
        predictions = self.detector(
            image,
            candidate_labels=self.CHEATING_CLASSES,
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
            
        import datetime
        return {
            "id": 2, 
            "timestamp": datetime.datetime.now().isoformat(),
            "probability": round(max_probability, 4) if max_probability > 0 else 0.0,
            "evidence": evidence_str            
        }

    @modal.asgi_app()
    def serve(self):
        return create_app()


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    fastapi_app = create_app()
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=120,
        h11_max_incomplete_event_size=104857600,
        factory=True,
    )