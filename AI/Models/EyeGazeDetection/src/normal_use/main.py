import cv2 as cv
import numpy as np
from collections import deque
<<<<<<< HEAD
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
=======
from Gaze import GazeDetector
from datetime import datetime

SMOOTHING_BUFFER_SIZE = 3
ENVELOPE_WINDOW       = 90

AWAY_SHORT_SECONDS = 3.0
AWAY_LONG_SECONDS  = 5.0


class GazeSession:

    def __init__(self):
        self._detector = GazeDetector()

        self._gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)

        self._envelope_x = deque(maxlen=ENVELOPE_WINDOW)

        self._initialized = False

        self._baseline_center_x = None
        self._baseline_std_x    = None

        self._attention_state = "INITIALIZING"
        self._away_start_time = None
        self._event_id        = 0

    def process_gaze_frame(self, frame):
        current_time = datetime.now().timestamp()

        h_ratio, _, face_present = self._detector.get_gaze_ratio(frame)

        if not face_present:
            self._attention_state = "NO_FACE"
            self._away_start_time = None
            return self._build_event("NO_FACE", 1.0, "NO_FACE")

        if not self._initialized:
            self._envelope_x.append(h_ratio)

            if len(self._envelope_x) >= ENVELOPE_WINDOW:
                self._initialized = True

            return self._build_event("INITIALIZING", 0.0, "CALIBRATING")

        if self._baseline_center_x is None:
            self._baseline_center_x = np.median(self._envelope_x)
            self._baseline_std_x    = np.std(self._envelope_x)

        self._gaze_history_x.append(h_ratio)
        avg_x = sum(self._gaze_history_x) / len(self._gaze_history_x)
        tolerance_x = max(0.07, self._baseline_std_x * 3.5)
        inside_safe_zone = abs(avg_x - self._baseline_center_x) <= tolerance_x

        if inside_safe_zone:
            self._attention_state = "ON_SCREEN"
            self._away_start_time = None
        else:
            if self._away_start_time is None:
                self._away_start_time = current_time

            elapsed = current_time - self._away_start_time

            if elapsed >= AWAY_LONG_SECONDS:
                self._attention_state = "AWAY_LONG"
            elif elapsed >= AWAY_SHORT_SECONDS:
                self._attention_state = "AWAY_SHORT"
            else:
                self._attention_state = "ON_SCREEN"

        if self._attention_state == "ON_SCREEN":
            probability = 0.0
        elif self._attention_state == "AWAY_SHORT":
            probability = 0.5
        else:   
            probability = 1.0

        return self._build_event(
            self._attention_state, probability, self._attention_state
        )

    def _build_event(self, state, probability, evidence):
        self._event_id = 1
        return {
            "id":              self._event_id,
            "timestamp":       datetime.now().isoformat(),
            "attention_state": state,
            "probability":     round(probability, 4),
            "evidence":        evidence,
        }


if __name__ == "__main__":
    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("Camera not detected.")
        exit()

    session = GazeSession()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (900, 600))

        event = session.process_gaze_frame(frame)
        print(event)

        color = (0, 255, 0)
        if event["attention_state"] in ["AWAY_LONG", "NO_FACE"]:
            color = (0, 0, 255)
        elif event["attention_state"] == "AWAY_SHORT":
            color = (0, 255, 255)

        cv.putText(
            frame,
            f'{event["attention_state"]} | P={event["probability"]} | ev={event["evidence"]}',
            (30, 50),
            cv.FONT_HERSHEY_PLAIN,
            2,
            color,
            2,
        )

        cv.imshow("Proctoring System - Attention Model", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
