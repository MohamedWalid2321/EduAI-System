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

    from routes import object_router, gaze_router, face_router, new_object_route
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


@app.cls(image=modal_image, gpu="any", scaledown_window=600)
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

        log.info("⏳ Loading Eye Gaze model...")
        import Models.EyeGazeDetection.src.Server.localMain  # noqa: F401
        log.info("✅ Eye Gaze loaded")

        log.info("⏳ Loading Face Recognition model (hybrid)...")
        from Models.Face_Recognition_Service import FaceRecognition  # noqa: F401
        FaceRecognition()
        log.info("✅ Face Recognition loaded (RetinaFace + ArcFace ONNX)")

        log.info("🚀 All models preloaded — container is warm!")

    @modal.asgi_app()
    def serve(self):
        return create_app()

@app.cls(
    image=modal_image,
    gpu="any",
    scaledown_window=600,
    min_containers=2,
    max_containers=10,
    memory=16384,
)
class GazeService:

    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root/app")
        sys.path.insert(0, "/root/app/Models/EyeGazeDetection/src/Server")
        from Models.EyeGazeDetection.src.Server.Gaze import GazeDetector as GD
        self._GazeDetector = GD
        self._detectors: dict[str, object] = {}
        print("[GazeService] Container ready.")

    @modal.fastapi_endpoint(method="POST")
    def detect(self, payload: dict) -> dict:
        """
        Accept a single frame and return gaze result.
        Payload: { "session_id": str, "frame": base64 }
        Returns: { "h_ratio": f, "v_ratio": f, "face_present": b }
        """
        import cv2, base64
        import numpy as np

        session_id = payload["session_id"]
        frame_b64  = payload["frame"]

        if session_id not in self._detectors:
            self._detectors[session_id] = self._GazeDetector()
            print(f"[GazeService] New detector for {session_id}")
        
        decode_start = time.time()
        img_bytes = base64.b64decode(frame_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decode_end = time.time()

        if frame is None:
            return {"h_ratio": 0.0, "v_ratio": 0.0, "face_present": False}
        
        process_start = time.time()
        h, v, face = self._detectors[session_id].get_gaze_ratio(frame)
        process_end = time.time()

        print(
                f"[Modal] session={session_id} | "
                f"decode={decode_end - decode_start:.3f}s | "
                f"mediapipe={process_end - process_start:.3f}s | "
                f"total={process_end - decode_start:.3f}s"
            )
        
        return {
            "h_ratio":      h,
            "v_ratio":      v,
            "face_present": face,
        }

    @modal.fastapi_endpoint(method="DELETE")
    def clear(self, payload: dict) -> dict:
        session_id = payload.get("session_id", "")
        self._detectors.pop(session_id, None)
        print(f"[GazeService] Cleared {session_id}")
        return {"cleared": True}


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