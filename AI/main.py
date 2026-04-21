"""
EduAI Proctoring API — FastAPI + Modal entrypoint.

Run locally  (from eye_gaze conda env):
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
import time

import modal

# Face enrollment vectors in Redis expire after 3 hours by default.
# Override by setting FACE_ENROLLMENT_TTL_SECONDS in the deployment environment.
os.environ.setdefault("FACE_ENROLLMENT_TTL_SECONDS", "10800")

# Modal secret used to inject Upstash Redis credentials at deploy/runtime.
# Create it with:
#   modal secret create eduai-upstash-redis \
#       UPSTASH_REDIS_REST_URL=... \
#       UPSTASH_REDIS_REST_TOKEN=...
_upstash_secret_name = os.getenv("MODAL_UPSTASH_SECRET_NAME", "eduai-upstash-redis")
_upstash_secret = modal.Secret.from_name(_upstash_secret_name)

# ---------------------------------------------------------------------------
# Path setup — ensure sub-packages can be imported
# ---------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
_server_path = os.path.join(
    _current_dir, "Models", "EyeGazeDetection", "src", "Server"
)
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


# ---------------------------------------------------------------------------
# Factory: builds the FastAPI app (heavy imports happen HERE, not at parse time)
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
        description="Unified AI proctoring — gaze, object detection & face recognition.",
        version="1.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from routes import speech_router, gaze_router, object_router, face_router, new_object_route
    application.include_router(speech_router)
    application.include_router(object_router)
    application.include_router(gaze_router)
    application.include_router(face_router)
    application.include_router(new_object_route)

    @application.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok"}

    return application


# ---------------------------------------------------------------------------
# Modal deployment
# ---------------------------------------------------------------------------
app = modal.App("eduai-proctoring")

modal_image = (
    modal.Image.debian_slim(python_version="3.10")
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
    gpu="L40S",
    scaledown_window=600,
    secrets=[_upstash_secret],
)
class Proctoring:
    """Modal class that keeps models warm in memory between requests."""

    @modal.enter()
    def preload(self):
        import sys as _sys

        if "/root/app" not in _sys.path:
            _sys.path.insert(0, "/root/app")
        _gaze_server = "/root/app/Models/EyeGazeDetection/src/Server"
        if _gaze_server not in _sys.path:
            _sys.path.insert(0, _gaze_server)

        log = logging.getLogger("preload")
        logging.basicConfig(level=logging.INFO)

        log.info("⏳ Loading YOLO model...")
        from Models.objectDetectionYolo.objectDetection import yoloDetect  # noqa: F401        
        log.info("✅ YOLO loaded")

        # log.info("⏳ Loading OWL-ViT model...")
        # from Models.objectDetectionOWL_VIT.main_detect import load_owl_vit  # noqa: F401
        # load_owl_vit()
        # log.info("✅ OWL-ViT loaded")

        #speech detection doesn't require a heavy model load, so we skip preloading it.

        log.info("⏳ Loading Eye Gaze model...")
        import Models.EyeGazeDetection.src.Server.localMain  # noqa: F401
        log.info("✅ Eye Gaze loaded")


        # log.info("⏳ Loading Face Detection model...")
        # from Models.FaceDetection.face_detection import FaceDetectionService  # noqa: F401
        # FaceDetectionService()
        # log.info("✅ Face Detection loaded")


        # log.info("⏳ Loading Face Anti-Spoofing model (MiniFASNetV2)...")
        # from Models.FaceAntiSpoofing.fas import FaceAntiSpoofingService  # noqa: F401
        # FaceAntiSpoofingService()
        # log.info("✅ Face Anti-Spoofing loaded (MiniFASNetV2)")


        log.info("⏳ Loading Face Recognition model (hybrid + FAS)...")
        from Models.Face_Recognition_Service import FaceRecognition  # noqa: F401
        # Instantiate once to warm up SCRDF + ArcFace ONNX + MiniFASNetV2
        FaceRecognition()
        log.info("✅ Face Recognition loaded (SCRFD + ArcFace ONNX + MiniFASNetV2 FAS)")

        log.info("🚀 All models preloaded — container is warm!")

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