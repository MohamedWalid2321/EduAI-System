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
    p_left   = np.array(landmarks[eye_points[0]])
    p_right  = np.array(landmarks[eye_points[1]])
    p_top    = np.array(landmarks[eye_points[2]])
    p_bottom = np.array(landmarks[eye_points[3]])
    p_iris   = np.array(landmarks[eye_points[4]])

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
    h, w, _ = frame.shape

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
