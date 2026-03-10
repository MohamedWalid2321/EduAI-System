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
        allow_origins=["*"],      # In production, specify allowed origins for security, ex: ["https://myproctoringapp.com"]    
        allow_credentials=True,   # Allow cookies/auth headers to be sent across origins (if needed)
        allow_methods=["*"],      # Allow all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],      # Allow all headers (or specify if you want to restrict), ex: ["Authorization", "Content-Type"]
    )

    # Register routers — each AI module has its own route file
    from routes import object_router, gaze_router, face_router, new_object_route
    application.include_router(object_router)
    application.include_router(gaze_router)
    application.include_router(face_router)
    application.include_router(new_object_route)

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
    .pip_install("redis") 
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
        import Models.EyeGazeDetection.src.Server.localMain # noqa: F401
        log.info("✅ Eye Gaze loaded")

        log.info("⏳ Loading Face Recognition model (hybrid + FAS)...")
        from Models.Face_Recognition_Service import FaceRecognition  # noqa: F401
        # Instantiate once to warm up RetinaFace + ArcFace ONNX + MiniFASNetV2
        FaceRecognition()
        log.info("✅ Face Recognition loaded (RetinaFace + ArcFace ONNX + MiniFASNetV2 FAS)")

        log.info("🚀 All models preloaded — container is warm!")

    @modal.asgi_app() #asgi: "ASGI" stands for Asynchronous Server Gateway Interface, which is a specification for building asynchronous web applications in Python. By using @modal.asgi_app(), we can create an ASGI-compatible application that can handle asynchronous requests and responses, making it suitable for high-performance web applications.
    def serve(self):
        """Return the FastAPI app — models are already in memory from preload()."""
        return create_app()

@app.cls(
    image=modal_image,
    gpu="any",
    scaledown_window=600,
    min_containers=2,          # always 2 warm containers ready
    max_containers=10,         # scale up to 10 under heavy load
    memory=16384,              # 16GB RAM per container
    secrets=[modal.Secret.from_name("redis-secret")],
)
class GazeService:

    @modal.enter()
    def load(self):
        import sys, os, redis
        sys.path.insert(0, "/root/app")
        sys.path.insert(0, "/root/app/Models/EyeGazeDetection/src/Server")

        from Models.EyeGazeDetection.src.Server.Gaze import GazeDetector as GD
        self._GazeDetector  = GD
        self._detectors: dict[str, object] = {}
        self._container_id  = os.urandom(8).hex()

        self._redis = redis.Redis(
            host            = os.environ["REDIS_HOST"],
            port            = int(os.environ["REDIS_PORT"]),
            password        = os.environ["REDIS_PASSWORD"],
            decode_responses = True,
            ssl             = True,
        )
        print(f"[GazeService] Container {self._container_id} ready.")

    @modal.fastapi_endpoint(method="POST")
    def detect(self, payload: dict) -> dict:
        import cv2, base64, os
        import numpy as np

        session_id = payload["session_id"]
        frame_b64  = payload["frame"]

        # check Redis — does this user already belong to another container?
        owner = self._redis.get(f"gaze:owner:{session_id}")

        if owner is None:
            # first time — claim this user for this container
            self._redis.set(
                f"gaze:owner:{session_id}",
                self._container_id,
                ex=7200,    # expire after 2 hours (exam duration)
            )
            owner = self._container_id

        if owner != self._container_id:
            # this user belongs to a different container
            # tell localMain to retry — Modal will eventually route correctly
            return {
                "redirect":     True,
                "h_ratio":      0.0,
                "v_ratio":      0.0,
                "face_present": False,
            }

        # this user belongs to THIS container — create detector if first request
        if session_id not in self._detectors:
            self._detectors[session_id] = self._GazeDetector()
            print(f"[GazeService] New detector for {session_id} in container {self._container_id}")

        # decode frame
        img_bytes = base64.b64decode(frame_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"redirect": False, "h_ratio": 0.0, "v_ratio": 0.0, "face_present": False}

        # use THIS user's dedicated detector — no overlap with any other user
        h, v, face = self._detectors[session_id].get_gaze_ratio(frame)

        return {
            "redirect":     False,
            "h_ratio":      h,
            "v_ratio":      v,
            "face_present": face,
        }

    @modal.fastapi_endpoint(method="DELETE")
    def clear(self, payload: dict) -> dict:
        session_id = payload.get("session_id", "")
        self._detectors.pop(session_id, None)
        self._redis.delete(f"gaze:owner:{session_id}")
        print(f"[GazeService] Cleared {session_id} from container {self._container_id}")
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
        port=8000 ,
        timeout_keep_alive=120,
        h11_max_incomplete_event_size=104857600,factory=True)
