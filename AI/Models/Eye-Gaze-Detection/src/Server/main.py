import numpy as np
from collections import deque
from datetime import datetime
from Gaze import get_gaze_ratio

# ===============================
# Persistent internal state
# ===============================
SMOOTHING_BUFFER_SIZE = 3
gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
gaze_history_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)

SUSPICIOUS_TIME_THRESHOLD = 45
suspicious_counter = 0

SAFE_DOWN_LIMIT = 0.65
BORDERLINE_DOWN_LIMIT = 0.80

cal_h_min, cal_h_max = 1.0, 0.0
cal_v_min, cal_v_max = 1.0, 0.0

EVENT_ID = 0


# ===============================
# Public function
# ===============================
def analyze_frame(frame, calibrating=False):
    """
    Analyze a single video frame and return gaze-based suspicion info.

    Args:
        frame (np.ndarray): BGR frame
        calibrating (bool): whether calibration is active

    Returns:
        dict: JSON-compatible result
    """
    global suspicious_counter
    global cal_h_min, cal_h_max, cal_v_min, cal_v_max
    global EVENT_ID

    timestamp = datetime.utcnow().isoformat()

    h_ratio, v_ratio = get_gaze_ratio(frame)

    # -------------------------------
    # Calibration mode
    # -------------------------------
    if calibrating:
        gaze_history_x.clear()
        gaze_history_y.clear()
        suspicious_counter = 0

        cal_h_min = min(cal_h_min, h_ratio)
        cal_h_max = max(cal_h_max, h_ratio)
        cal_v_min = min(cal_v_min, v_ratio)
        cal_v_max = max(cal_v_max, v_ratio)

        return {
            "id": None,
            "timestamp": timestamp,
            "flag": False,
            "probability": 0.0,
            "evidence": {
                "state": "CALIBRATING"
            }
        }

    # -------------------------------
    # Normalize gaze
    # -------------------------------
    raw_x = (h_ratio - cal_h_min) / (cal_h_max - cal_h_min) if cal_h_max > cal_h_min else 0.5
    raw_y = (v_ratio - cal_v_min) / (cal_v_max - cal_v_min) if cal_v_max > cal_v_min else 0.5

    raw_x = np.clip(raw_x, 0.0, 1.0)
    raw_y = np.clip(raw_y, 0.0, 1.0)

    gaze_history_x.append(raw_x)
    gaze_history_y.append(raw_y)

    avg_x = sum(gaze_history_x) / len(gaze_history_x)
    avg_y = sum(gaze_history_y) / len(gaze_history_y)

    # -------------------------------
    # Horizontal direction
    # -------------------------------
    horizontal = "CENTER"
    if avg_x < 0.25:
        horizontal = "LEFT"
    elif avg_x > 0.75:
        horizontal = "RIGHT"

    # -------------------------------
    # Vertical semantic zone
    # -------------------------------
    if avg_y <= SAFE_DOWN_LIMIT:
        vertical = "SAFE"
    elif avg_y <= BORDERLINE_DOWN_LIMIT:
        vertical = "BORDERLINE"
    else:
        vertical = "EXTREME"

    # -------------------------------
    # Suspicion logic
    # -------------------------------
    if horizontal in ("LEFT", "RIGHT"):
        suspicious_counter += 1
    elif vertical == "SAFE":
        suspicious_counter = max(0, suspicious_counter - 2)
    elif vertical == "BORDERLINE":
        suspicious_counter += 0.3
    else:  # EXTREME
        suspicious_counter += 2

    probability = min(1.0, suspicious_counter / SUSPICIOUS_TIME_THRESHOLD)
    flag = suspicious_counter > SUSPICIOUS_TIME_THRESHOLD

    if flag:
        EVENT_ID += 1

    # -------------------------------
    # JSON output
    # -------------------------------
    return {
        "id": EVENT_ID if flag else None,
        "timestamp": timestamp,
        "flag": flag,
        "probability": round(probability, 4),
        "evidence": {
            "horizontal": horizontal,
            "vertical": vertical,
            "avg_x": round(avg_x, 3),
            "avg_y": round(avg_y, 3)
        }
    }