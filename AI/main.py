"""
EduAI Proctoring API — FastAPI + Modal entrypoint.

Run locally  (from eye_gaze conda env):
    conda activate eye_gaze
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

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
    """Build and return the FastAPI application.

    All AI-model imports are deferred into this function so that
    `modal serve / deploy` can parse this file without needing
    TensorFlow, mediapipe, etc. on the local machine.
    """
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
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Router import triggers AI model loading — only happens inside create_app()
    from routes.modal_route import router as analysis_router
    application.include_router(analysis_router)

    @application.get("/health", tags=["Health"])
    async def health_check():
        """Simple liveness probe."""
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
        ignore=["__pycache__", ".git", ".venv"],
    )
)


@app.cls(image=modal_image, gpu="any", container_idle_timeout=600)
class Proctoring:
    """Modal class that keeps models warm in memory between requests."""

    @modal.enter()
    def preload(self):
        """Runs ONCE when the container starts — loads all AI models."""
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

        log.info("⏳ Loading Eye Gaze model...")
        import Models.EyeGazeDetection.src.Server.localMain  # noqa: F401
        log.info("✅ Eye Gaze loaded")

        log.info("⏳ Loading Face Recognition model...")
        from Models.Face_Recognition_Service import FaceRecognition  # noqa: F401
        log.info("✅ Face Recognition loaded")

        log.info("🚀 All models preloaded — container is warm!")

    @modal.asgi_app() #asgi: "ASGI" stands for Asynchronous Server Gateway Interface, which is a specification for building asynchronous web applications in Python. By using @modal.asgi_app(), we can create an ASGI-compatible application that can handle asynchronous requests and responses, making it suitable for high-performance web applications.
    def serve(self):
        """Return the FastAPI app — models are already in memory from preload()."""
        return create_app()


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    fastapi_app = create_app()
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
