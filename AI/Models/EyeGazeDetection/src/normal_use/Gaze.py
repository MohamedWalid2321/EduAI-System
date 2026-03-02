<<<<<<< HEAD
#dynamic calibration version

import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh with lower confidence for partial face detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,  # reduced to allow partial face
    min_tracking_confidence=0.3
)

def get_euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def normalize_lighting(eye_region):
    gray = cv.cvtColor(eye_region, cv.COLOR_BGR2GRAY)
    return cv.equalizeHist(gray)

def get_head_pose(landmarks):
    """Optional: returns yaw and pitch offsets based on nose and eyes."""
    left_corner = np.array(landmarks[33])
    right_corner = np.array(landmarks[263])
    nose_tip = np.array(landmarks[1])
    
    yaw = (nose_tip[0] - (left_corner[0] + right_corner[0]) / 2) / max(1, right_corner[0] - left_corner[0])
    pitch = (nose_tip[1] - (np.array(landmarks[10])[1] + np.array(landmarks[152])[1]) / 2) / max(1, np.array(landmarks[152])[1] - np.array(landmarks[10])[1])
    
    return yaw, pitch

def calculate_eye_ratio(frame, landmarks, eye_points):
    # Landmarks
=======
import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import time

FACE_LANDMARKER_MODEL_PATH     = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "face_landmarker", 
    "face_landmarker.task"
)
LANDMARKS_CACHE_EXPIRE_SECONDS = 1.5

def _amplify_nonlinear(ratio, strength):
    x = np.tanh((ratio - 0.5) * strength)
    return float(0.5 + x / 2)

def _rotation_matrix_to_pitch_yaw(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
    return float(pitch), float(yaw)

def _calculate_eye_ratio(landmarks, eye_points):
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
    p_left   = np.array(landmarks[eye_points[0]])
    p_right  = np.array(landmarks[eye_points[1]])
    p_top    = np.array(landmarks[eye_points[2]])
    p_bottom = np.array(landmarks[eye_points[3]])
    p_iris   = np.array(landmarks[eye_points[4]])

<<<<<<< HEAD
    # Eye bounding box (optional debug)
    x_min = max(0, min(p_left[0], p_right[0], p_iris[0]) - 3)
    y_min = max(0, min(p_top[1], p_bottom[1], p_iris[1]) - 3)
    x_max = min(frame.shape[1], max(p_left[0], p_right[0], p_iris[0]) + 3)
    y_max = min(frame.shape[0], max(p_top[1], p_bottom[1], p_iris[1]) + 3)

    eye_region = frame[y_min:y_max, x_min:x_max]
    _ = normalize_lighting(eye_region)

    # --- Horizontal ratio (corrected for inversion) ---
    eye_width = max(1, p_right[0] - p_left[0])
    h_ratio = (p_right[0] - p_iris[0]) / eye_width  # swapped numerator

    # --- Vertical ratio (corrected) ---
    eye_height = max(1, p_bottom[1] - p_top[1])
    v_ratio = (p_bottom[1] - p_iris[1]) / eye_height  # swapped numerator

    # --- Amplify small movements ---
    h_ratio = np.clip(0.5 + (h_ratio - 0.5) * 1.8, 0.0, 1.0)
    v_ratio = np.clip(0.5 + (v_ratio - 0.5) * 1.8, 0.0, 1.0)

    return h_ratio, v_ratio

last_landmarks = None  # cache for partial face detection

def get_gaze_ratio(frame):
    global last_landmarks
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape # frame contain height and width and channel but we need height and width

    avg_h_ratio, avg_v_ratio = 0.5, 0.5  # default CENTER

    if results.multi_face_landmarks:
        mp_points = np.array([
            np.multiply([p.x, p.y], [w, h]).astype(int)
            for p in results.multi_face_landmarks[0].landmark
        ])
        last_landmarks = mp_points  # update cache
    elif last_landmarks is not None:
        mp_points = last_landmarks  # use last known landmarks
    else:
        return avg_h_ratio, avg_v_ratio  # no face

    left_eye_idx  = [33, 133, 159, 145, 468]
    right_eye_idx = [362, 263, 386, 374, 473]

    h_l, v_l = calculate_eye_ratio(frame, mp_points, left_eye_idx)
    h_r, v_r = calculate_eye_ratio(frame, mp_points, right_eye_idx)

    avg_h_ratio = (h_l + h_r) / 2
    avg_v_ratio = (v_l + v_r) / 2

    # Optional head-pose compensation
    yaw, pitch = get_head_pose(mp_points)
    avg_h_ratio -= yaw * 0.25
    avg_v_ratio += pitch * 0.25

    avg_h_ratio = np.clip(avg_h_ratio, 0.0, 1.0)
    avg_v_ratio = np.clip(avg_v_ratio, 0.0, 1.0)

    return avg_h_ratio, avg_v_ratio
=======
    eye_w = max(1, p_right[0] - p_left[0])
    eye_h = max(1, p_bottom[1] - p_top[1])

    h_ratio = (p_iris[0] - p_left[0]) / eye_w
    v_ratio = 1.0 - (p_bottom[1] - p_iris[1]) / eye_h

    return _amplify_nonlinear(h_ratio, 2.0), _amplify_nonlinear(v_ratio, 1.5)

class GazeDetector:

    def __init__(self):
        self._last_landmarks      = None
        self._last_landmarks_time = None
        self._last_pitch_deg      = 0.0
        self._last_yaw_deg        = 0.0
        self._last_timestamp_ms = 0

        self._landmarker     = None
        self._use_landmarker = False
        self._init_landmarker()

    def _init_landmarker(self):
        if not os.path.isfile(FACE_LANDMARKER_MODEL_PATH):
            print(
                f"[GazeDetector] WARNING: model file '{FACE_LANDMARKER_MODEL_PATH}' not found."
            )
            return
        
        #Read the model file into memory as raw bytes
        with open(FACE_LANDMARKER_MODEL_PATH, "rb") as f:
            model_data = f.read()

        #
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
        self._landmarker     = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._use_landmarker = True
        print("[GazeDetector] FaceLandmarker ready.")

    def get_gaze_ratio(self, frame):
        if not self._use_landmarker:
            return 0.0, 0.0, False

        h, w = frame.shape[:2]
        rgb  = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_img       = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.time() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result       = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        if result.face_landmarks:
            face1 = result.face_landmarks[0]
            mp_points = np.array([
                [int(p.x * w), int(p.y * h)]
                for p in face1
            ])
            self._last_landmarks = mp_points
            self._last_landmarks_time = time.time()
            face_detect = True
        elif (
            self._last_landmarks is not None and
            self._last_landmarks_time is not None and
            (time.time() - self._last_landmarks_time) < LANDMARKS_CACHE_EXPIRE_SECONDS
        ):
            mp_points   = self._last_landmarks
            face_detect = True
        else:
            self._last_landmarks = None
            return 0.0, 0.0, False

        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0])
            pitch_deg, yaw_deg   = _rotation_matrix_to_pitch_yaw(mat[:3, :3])
            self._last_pitch_deg = pitch_deg
            self._last_yaw_deg   = yaw_deg
        else:
            pitch_deg = self._last_pitch_deg
            yaw_deg   = self._last_yaw_deg

        left_idx  = [33,  133, 159, 145, 468]
        right_idx = [362, 263, 386, 374, 473]

        h_l, v_l = _calculate_eye_ratio(mp_points, left_idx)
        h_r, v_r = _calculate_eye_ratio(mp_points, right_idx)

        avg_h = (h_l + h_r) / 2
        avg_v = (v_l + v_r) / 2

        avg_h += (yaw_deg / 90.0) * 0.15
        avg_h  = float(np.clip(avg_h, 0.0, 1.0))
        avg_v  = float(np.clip(avg_v, 0.0, 1.0))

        return avg_h, avg_v, face_detect
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
