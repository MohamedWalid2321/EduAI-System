import cv2 as cv
import numpy as np
from collections import deque
from Gaze import get_gaze_ratio
from datetime import datetime

capture = cv.VideoCapture(0)

# --- CONFIGURATION ---
# Phase 1: Smoothing variables
SMOOTHING_BUFFER_SIZE = 6  # Higher = smoother but more delay
gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
gaze_history_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# Phase 2: Suspicious Logic variables
# If FPS is ~30, then 45 frames is roughly 1.5 seconds of looking away
SUSPICIOUS_TIME_THRESHOLD = 45 
suspicious_counter = 0

# Calibration Defaults
cal_h_min, cal_h_max = 1.0, 0.0
cal_v_min, cal_v_max = 1.0, 0.0

if not capture.isOpened():
    print("Camera not detected.")
    exit()

while True:
    ret, frame = capture.read()
    if not ret: break
    
    frame = cv.flip(frame, 1)
    height, width, _ = frame.shape

    # 1. Get RAW Gaze Ratios (from your media.py)
    h_ratio, v_ratio = get_gaze_ratio(frame)

    # --- SECURITY CHECK: IS THE USER GONE? ---
    # In media.py, if no face is found, it returns exactly 0.5 for both.
    # It is extremely rare to look at EXACTLY 0.500000 naturally.
    if h_ratio == 0.5 and v_ratio == 0.5:
        cv.putText(frame, "NO FACE DETECTED", (50, height // 2), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv.rectangle(frame, (0,0), (width, height), (0,0,255), 20)
        # Force suspicious counter up so they can't cheat by hiding
        suspicious_counter += 2

    key = cv.waitKey(1) & 0xFF
    is_calibrating = (key == ord('c'))

    if is_calibrating:
        # Reset behavior logic during calibration
        suspicious_counter = 0
        gaze_history_x.clear()
        gaze_history_y.clear()

        cv.putText(frame, "CALIBRATING... Look at corners!", (20, 50), 
                   cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        
        if h_ratio < cal_h_min: cal_h_min = h_ratio
        if h_ratio > cal_h_max: cal_h_max = h_ratio
        if v_ratio < cal_v_min: cal_v_min = v_ratio
        if v_ratio > cal_v_max: cal_v_max = v_ratio
        
        # Display Raw Stats
        cv.putText(frame, f"H: {cal_h_min:.2f}-{cal_h_max:.2f}", (20, 100), cv.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)
        cv.putText(frame, f"V: {cal_v_min:.2f}-{cal_v_max:.2f}", (20, 130), cv.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)

    else:
        # --- MAPPING RAW RATIOS TO SCREEN COORDINATES ---
        h_denom = (cal_h_max - cal_h_min)
        v_denom = (cal_v_max - cal_v_min)
        
        raw_x, raw_y = 0.5, 0.5

        if h_denom != 0: raw_x = (h_ratio - cal_h_min) / h_denom
        if v_denom != 0: raw_y = (v_ratio - cal_v_min) / v_denom

        # Clamp
        raw_x = np.clip(raw_x, 0.0, 1.0)
        raw_y = np.clip(raw_y, 0.0, 1.0)

        # --- PHASE 1: JITTER REDUCTION (SMOOTHING) ---
        gaze_history_x.append(raw_x)
        gaze_history_y.append(raw_y)

        # Calculate average of the history buffer
        avg_x = sum(gaze_history_x) / len(gaze_history_x)
        avg_y = sum(gaze_history_y) / len(gaze_history_y)

        # --- DIRECTION LOGIC ---
        text_h = "CENTER"
        if avg_x < 0.20: text_h = "LEFT"    # Was 0.35
        elif avg_x > 0.80: text_h = "RIGHT" # Was 0.65

        text_v = "CENTER"
        if avg_y < 0.20: text_v = "UP"      # Was 0.35
        elif avg_y > 0.80: text_v = "DOWN"  # Was 0.65

        # --- PHASE 2: SUSPICIOUS BEHAVIOR DETECTION ---
        # If looking anywhere other than CENTER, increase counter
        if text_h != "CENTER" or text_v != "CENTER":
            suspicious_counter += 1
        else:
            # If they look back to center, decrease counter (grace period) or reset
            if suspicious_counter > 0:
                suspicious_counter -= 2 

        # Check Threshold
        is_suspicious = suspicious_counter > SUSPICIOUS_TIME_THRESHOLD

        # --- VISUALIZATION ---
        color = (0, 255, 0) # Green by default

        if is_suspicious:
            color = (0, 0, 255) # Red
            cv.putText(frame, "SUSPICIOUS ACTIVITY: LOOKING AWAY", (50, height // 2), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Draw red border
            cv.rectangle(frame, (0,0), (width, height), (0,0,255), 10)

            # NEW: Log the event to a file
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open("cheating_log.txt", "a") as f:
                f.write(f"[{timestamp}] SUSPICIOUS ACTIVITY DETECTED\n")

            # NEW: Take a photo every 50 frames (to avoid filling storage)
            if suspicious_counter % 50 == 0:
                filename = f"evidence_{datetime.now().strftime('%H%M%S')}.jpg"
                cv.imwrite(filename, frame)
                print(f"Evidence saved: {filename}")    
        
        elif suspicious_counter > 10: 
            # Warning State (Orange)
            color = (0, 255, 255) 
            cv.putText(frame, "Warning: Focus...", (50, height - 50), 
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # Display Direction
        cv.putText(frame, f"{text_h} - {text_v}", (30, 50), cv.FONT_HERSHEY_PLAIN, 2, color, 2)
        
        # Draw Mini-map (Smoothed)
        map_x = int(avg_x * 200) + 400
        map_y = int(avg_y * 150) + 300
        cv.rectangle(frame, (400, 300), (600, 450), (255, 255, 255), 1)
        # Draw the target dot
        cv.circle(frame, (map_x, map_y), 8, color, -1)

    cv.imshow("Proctoring System - Eye Module", frame)

    if key == ord('q'):
        break

capture.release()
cv.destroyAllWindows()