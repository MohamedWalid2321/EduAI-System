import cv2 as cv
import numpy as np
from collections import deque
from Gaze import get_gaze_ratio
from datetime import datetime

# ==============================
# STATE (PERSISTENT)
# ==============================
SMOOTHING_BUFFER_SIZE = 3
gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
gaze_history_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)

SUSPICIOUS_TIME_THRESHOLD = 30
suspicious_counter = 0
event_id = 0

SAFE_DOWN_LIMIT = 0.55
BORDERLINE_DOWN_LIMIT = 0.70


cal_h_min, cal_h_max = 1.0, 0.0
cal_v_min, cal_v_max = 1.0, 0.0

# ==============================
# CORE FUNCTION (FOR main.py)
# ==============================
def process_gaze_frame(frame, calibrating=False):
    global suspicious_counter, cal_h_min, cal_h_max
    global cal_v_min, cal_v_max, event_id

    h_ratio, v_ratio = get_gaze_ratio(frame)

    # --- No face detected ---
    if (h_ratio == 0.5 and v_ratio == 0.5) and not calibrating:
        suspicious_counter += 2
        return _build_event(
            flag=True,
            probability=1.0,
            evidence="NO FACE DETECTED"
        )

    # --- Calibration ---
    if calibrating:
        suspicious_counter = 0
        gaze_history_x.clear()
        gaze_history_y.clear()

        cal_h_min = min(cal_h_min, h_ratio)
        cal_h_max = max(cal_h_max, h_ratio)
        cal_v_min = min(cal_v_min, v_ratio)
        cal_v_max = max(cal_v_max, v_ratio)

        return _build_event(
            flag=False,
            probability=0.0,
            evidence="CALIBRATING"
        )

    # --- Normalize ---
    h_denom = cal_h_max - cal_h_min
    v_denom = cal_v_max - cal_v_min

    raw_x = (h_ratio - cal_h_min) / h_denom if h_denom else 0.5
    raw_y = (v_ratio - cal_v_min) / v_denom if v_denom else 0.5

    raw_x = np.clip(raw_x, 0.0, 1.0)
    raw_y = np.clip(raw_y, 0.0, 1.0)

    gaze_history_x.append(raw_x)
    gaze_history_y.append(raw_y)

    avg_x = sum(gaze_history_x) / len(gaze_history_x)
    avg_y = sum(gaze_history_y) / len(gaze_history_y)

    # --- Horizontal ---
    if avg_x < 0.25:
        text_h = "LEFT"
    elif avg_x > 0.75:
        text_h = "RIGHT"
    else:
        text_h = "CENTER"

    # --- Vertical ---
    if avg_y <= SAFE_DOWN_LIMIT:
        vertical_state = "SAFE"
    elif avg_y <= BORDERLINE_DOWN_LIMIT:
        vertical_state = "BORDERLINE"
    else:
        vertical_state = "EXTREME"

    # --- Suspicion logic ---
    if text_h in ["LEFT", "RIGHT"]:
        suspicious_counter += 1
    elif vertical_state == "SAFE":
        suspicious_counter = max(0, suspicious_counter - 2)
    elif vertical_state == "BORDERLINE":
        suspicious_counter += 0.3
    elif vertical_state == "EXTREME":
        suspicious_counter += 2

    probability = min(1.0, suspicious_counter / SUSPICIOUS_TIME_THRESHOLD)
    flag = suspicious_counter > SUSPICIOUS_TIME_THRESHOLD

    return _build_event(
        flag=flag,
        probability=probability,
        evidence=f"{text_h} | DOWN:{vertical_state}"
    )


# ==============================
# EVENT BUILDER
# ==============================
def _build_event(flag, probability, evidence):
    global event_id
    event_id = 1

    return {
        "id": event_id,
        "timestamp": datetime.now().isoformat(),
        "flag": flag,
        "probability": round(probability, 4),
        "evidence": evidence
    }


# ==============================
# LOCAL TESTING (OPTIONAL)
# ==============================
if __name__ == "__main__":
    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("Camera not detected.")
        exit()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        key = cv.waitKey(1) & 0xFF
        is_calibrating = (key == ord('c'))

        event = process_gaze_frame(frame, is_calibrating)
        print(event)
        color = (0, 255, 0)
        if event["flag"]:
            color = (0, 0, 255)

        cv.putText(
            frame,
            f'{event["evidence"]} | P={event["probability"]}',
            (30, 50),
            cv.FONT_HERSHEY_PLAIN,
            2,
            color,
            2
        )

        cv.imshow("Proctoring System - Eye Module", frame)

        if key == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()