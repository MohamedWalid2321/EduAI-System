import cv2 as cv
import numpy as np
from collections import deque
from Gaze import get_gaze_ratio
from datetime import datetime

capture = cv.VideoCapture(0)

# --- Smoothing ---
SMOOTHING_BUFFER_SIZE = 3
gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
gaze_history_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# --- Suspicion ---
SUSPICIOUS_TIME_THRESHOLD = 45
suspicious_counter = 0

# --- Vertical zones ---
SAFE_DOWN_LIMIT = 0.65
BORDERLINE_DOWN_LIMIT = 0.80

# --- Calibration ---
cal_h_min, cal_h_max = 1.0, 0.0
cal_v_min, cal_v_max = 1.0, 0.0

if not capture.isOpened():
    print("Camera not detected.")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    height, width, _ = frame.shape

    h_ratio, v_ratio = get_gaze_ratio(frame)

    key = cv.waitKey(1) & 0xFF
    is_calibrating = (key == ord('c'))

    # --- No face ---
    no_face_detected = (h_ratio == 0.5 and v_ratio == 0.5) and not is_calibrating
    if no_face_detected:
        suspicious_counter += 2
        cv.putText(frame, "NO FACE DETECTED", (50, height // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 20)

    # --- Calibration ---
    if is_calibrating:
        suspicious_counter = 0
        gaze_history_x.clear()
        gaze_history_y.clear()

        cv.putText(frame, "CALIBRATING... Look around naturally",
                   (20, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cal_h_min = min(cal_h_min, h_ratio)
        cal_h_max = max(cal_h_max, h_ratio)
        cal_v_min = min(cal_v_min, v_ratio)
        cal_v_max = max(cal_v_max, v_ratio)

    else:
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

        # --- Horizontal direction ---
        text_h = "CENTER"
        if avg_x < 0.25:
            text_h = "LEFT"
        elif avg_x > 0.75:
            text_h = "RIGHT"

        # --- Vertical semantic zones ---
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

        is_suspicious = suspicious_counter > SUSPICIOUS_TIME_THRESHOLD
        color = (0, 255, 0)

        # --- Alerts ---
        if is_suspicious:
            color = (0, 0, 255)
            cv.putText(frame, "SUSPICIOUS ACTIVITY",
                       (50, height // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

            timestamp = datetime.now().strftime("%H:%M:%S")
            with open("cheating_log.txt", "a") as f:
                f.write(f"[{timestamp}] Suspicious behavior detected\n")

            if int(suspicious_counter) % 50 == 0:
                filename = f"evidence_{datetime.now().strftime('%H%M%S')}.jpg"
                cv.imwrite(filename, frame)

        elif suspicious_counter > 10:
            color = (0, 255, 255)
            cv.putText(frame, "Warning: Stay focused",
                       (50, height - 50),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # --- UI ---
        cv.putText(frame,
                   f"{text_h} | DOWN:{vertical_state}",
                   (30, 50),
                   cv.FONT_HERSHEY_PLAIN, 2, color, 2)

        cv.rectangle(frame, (400, 300), (600, 450), (255, 255, 255), 1)
        cv.circle(frame,
                  (int(avg_x * 200) + 400, int(avg_y * 150) + 300),
                  8, color, -1)

    cv.imshow("Proctoring System - Eye Module", frame)

    if key == ord('q'):
        break

capture.release()
cv.destroyAllWindows()