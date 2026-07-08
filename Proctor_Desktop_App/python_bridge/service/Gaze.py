import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker",
    "face_landmarker.task",
)
_LEGACY_SERVER_FACE_LANDMARKER_MODEL_PATH = os.path.join(
    _PROJECT_ROOT,
    "AI",
    "Models",
    "EyeGazeDetection",
    "src",
    "Server",
    "face_landmarker",
    "face_landmarker.task",
)
_LEGACY_SRC_FACE_LANDMARKER_MODEL_PATH = os.path.join(
    _PROJECT_ROOT,
    "AI",
    "Models",
    "EyeGazeDetection",
    "src",
    "face_landmarker",
    "face_landmarker.task",
)


def _resolve_face_landmarker_model_path() -> str:
    env_path = os.environ.get("FACE_LANDMARKER_MODEL_PATH")
    candidates = [
        env_path,
        _DEFAULT_FACE_LANDMARKER_MODEL_PATH,
        _LEGACY_SERVER_FACE_LANDMARKER_MODEL_PATH,
        _LEGACY_SRC_FACE_LANDMARKER_MODEL_PATH,
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return env_path or _DEFAULT_FACE_LANDMARKER_MODEL_PATH


FACE_LANDMARKER_MODEL_PATH = _resolve_face_landmarker_model_path()

LANDMARKS_CACHE_EXPIRE_SECONDS = 1.5


def _amplify_nonlinear(ratio, strength):
    x = np.tanh((ratio - 0.5) * strength)
    return float(0.5 + x / 2)


def _rotation_matrix_to_pitch_yaw(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    return float(pitch), float(yaw)


def _calculate_eye_ratio(landmarks, eye_points):
    p_left = np.array(landmarks[eye_points[0]], dtype=np.float64)
    p_right = np.array(landmarks[eye_points[1]], dtype=np.float64)
    p_top = np.array(landmarks[eye_points[2]], dtype=np.float64)
    p_bottom = np.array(landmarks[eye_points[3]], dtype=np.float64)
    p_iris = np.array(landmarks[eye_points[4]], dtype=np.float64)

    eye_w = max(1.0, p_right[0] - p_left[0])
    eye_h = max(1.0, p_bottom[1] - p_top[1])

    h_ratio = (p_iris[0] - p_left[0]) / eye_w
    v_ratio = 1.0 - (p_bottom[1] - p_iris[1]) / eye_h

    return _amplify_nonlinear(h_ratio, 2.5), _amplify_nonlinear(v_ratio, 2.5)


class GazeDetector:
    """
    Stateless gaze detector - runs on Modal.
    Each instance loads MediaPipe once (via @modal.enter) and is reused
    across requests within the same container.
    Returns only 3 values: h_ratio, v_ratio, face_present.
    All calibration and state logic lives in localMain.py on the desktop.
    """

    def __init__(self):
        self._last_landmarks = None
        self._last_landmarks_time = None
        self._last_pitch_deg = 0.0
        self._last_yaw_deg = 0.0
        self._last_timestamp_ms = 0

        self._landmarker = None
        self._use_landmarker = False
        self._init_landmarker()

        self.last_pitch_deg: float = 0.0
        self.last_yaw_deg: float = 0.0

    def _init_landmarker(self):
        if not os.path.isfile(FACE_LANDMARKER_MODEL_PATH):
            print(
                f"[GazeDetector] WARNING: model file '{FACE_LANDMARKER_MODEL_PATH}' not found."
            )
            return

        with open(FACE_LANDMARKER_MODEL_PATH, "rb") as f:
            model_data = f.read()

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._use_landmarker = True
        print("[GazeDetector] FaceLandmarker ready.")

    def get_gaze_ratio(self, frame) -> tuple[float, float, bool]:
        """
        Process a single BGR frame.

        Returns
        -------
        h_ratio      : float  0.0 (far left) -> 1.0 (far right)
        v_ratio      : float  0.0 (far down) -> 1.0 (far up)
        face_present : bool
        """
        if not self._use_landmarker:
            return 0.0, 0.0, False

        h, w = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.time() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        if result.face_landmarks:
            face1 = result.face_landmarks[0]
            mp_points = np.array(
                [[int(p.x * w), int(p.y * h)] for p in face1],
                dtype=np.float64,
            )
            self._last_landmarks = mp_points
            self._last_landmarks_time = time.time()
            face_detect = True
        elif (
            self._last_landmarks is not None
            and self._last_landmarks_time is not None
            and (time.time() - self._last_landmarks_time) < LANDMARKS_CACHE_EXPIRE_SECONDS
        ):
            mp_points = self._last_landmarks
            face_detect = True
        else:
            self._last_landmarks = None
            return 0.0, 0.0, False

        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0])
            pitch_deg, yaw_deg = _rotation_matrix_to_pitch_yaw(mat[:3, :3])
            self._last_pitch_deg = pitch_deg
            self._last_yaw_deg = yaw_deg
        else:
            pitch_deg = self._last_pitch_deg
            yaw_deg = self._last_yaw_deg

        self.last_pitch_deg = pitch_deg
        self.last_yaw_deg = yaw_deg

        left_idx = [33, 133, 159, 145, 468]
        right_idx = [362, 263, 386, 374, 473]

        h_l, v_l = _calculate_eye_ratio(mp_points, left_idx)
        h_r, v_r = _calculate_eye_ratio(mp_points, right_idx)

        avg_h = (h_l + h_r) / 2
        avg_v = (v_l + v_r) / 2

        # Yaw compensation: head turning shifts iris geometrically.
        # Subtract the yaw-induced shift so only true eye movement triggers.
        # Factor 0.30 ≈ correction per 90° of head turn — tune up if
        # head movement still triggers, tune down if real gaze is suppressed.
        avg_h -= (yaw_deg / 90.0) * 0.30
        avg_v -= (pitch_deg / 60.0) * 0.50

        avg_h = float(np.clip(avg_h, 0.0, 1.0))
        avg_v = float(np.clip(avg_v, 0.0, 1.0))

        return avg_h, 1 - avg_v, face_detect
