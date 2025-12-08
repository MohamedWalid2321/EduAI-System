import cv2 as cv
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_eye_ratio(frame, landmarks, eye_points):
    # Extract coordinates
    p_left   = landmarks[eye_points[0]]
    p_right  = landmarks[eye_points[1]]
    p_top    = landmarks[eye_points[2]]
    p_bottom = landmarks[eye_points[3]]
    p_iris   = landmarks[eye_points[4]]

    # Debug Visuals (Optional: Comment out for speed)
    cv.circle(frame, p_left,   2, (0, 255, 0), -1)
    cv.circle(frame, p_right,  2, (0, 255, 0), -1)
    cv.circle(frame, p_top,    2, (255, 255, 0), -1)
    cv.circle(frame, p_bottom, 2, (255, 255, 0), -1)
    cv.circle(frame, p_iris,   3, (0, 0, 255), -1)

    # --- HORIZONTAL RATIO ---
    dist_left_to_iris = get_euclidean_distance(p_left, p_iris)
    total_width       = get_euclidean_distance(p_left, p_right)
    
    h_ratio = 0.5
    if total_width != 0:
        h_ratio = dist_left_to_iris / total_width

    # --- VERTICAL RATIO ---
    dist_top_to_iris = get_euclidean_distance(p_top, p_iris)
    total_height     = get_euclidean_distance(p_top, p_bottom)
    
    v_ratio = 0.5
    if total_height != 0:
        v_ratio = dist_top_to_iris / total_height
        
    return h_ratio, v_ratio

def get_gaze_ratio(frame):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
    
    # Default values if no face found
    avg_h_ratio, avg_v_ratio = 0.5, 0.5

    if results.multi_face_landmarks:
        mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) 
                                for p in results.multi_face_landmarks[0].landmark])

        # Indices: [Left, Right, Top, Bottom, Iris]
        left_eye_indices = [33, 133, 159, 145, 468] 
        right_eye_indices = [362, 263, 386, 374, 473]

        h_ratio_left, v_ratio_left = calculate_eye_ratio(frame, mesh_points, left_eye_indices)
        h_ratio_right, v_ratio_right = calculate_eye_ratio(frame, mesh_points, right_eye_indices)

        avg_h_ratio = (h_ratio_left + h_ratio_right) / 2
        avg_v_ratio = (v_ratio_left + v_ratio_right) / 2

    return avg_h_ratio, avg_v_ratio