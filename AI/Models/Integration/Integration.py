# integration.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from datetime import datetime
from collections import deque
import numpy as np

# --- Import logic from other modules ---
from ObjectDetection import load_model, detect_objects
from GazeTest import get_gaze_ratio  # assumes GazeTest.py contains the function

# --- Configuration ---
SMOOTHING_BUFFER_SIZE = 3
SUSPICIOUS_TIME_THRESHOLD = 45
CALIBRATION_DURATION = 15  # seconds

# --- Load YOLO model ---
model = load_model("D:\\Downloads\\GradPro\\AI\\Models\\objectDetectionYolo\\best.pt")

# --- Eye gaze smoothing & calibration ---
gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
gaze_history_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)
suspicious_counter = 0
cal_h_min, cal_h_max = 1.0, 0.0
cal_v_min, cal_v_max = 1.0, 0.0
calibration_start_time = datetime.now()

# --- GUI ---
root = tk.Tk()
root.title("AI Proctoring System - Integrated Proctoring")
root.geometry("1000x700")

title_lbl = tk.Label(root, text="Exam Session In Progress", font=("Helvetica", 24, "bold"))
title_lbl.pack(pady=10)

video_frame = tk.Label(root)
video_frame.pack()

status_panel = tk.Frame(root, bg="#f0f0f0", pady=20)
status_panel.pack(fill="x", side="bottom")

status_lbl = tk.Label(status_panel, text="STATUS: CLEAN", font=("Arial", 20, "bold"), fg="green", bg="#f0f0f0")
status_lbl.pack()

info_lbl = tk.Label(status_panel, text="No anomalies detected.", font=("Arial", 12), bg="#f0f0f0")
info_lbl.pack()

btn_quit = ttk.Button(root, text="End Exam", command=root.destroy)
btn_quit.place(x=900, y=20)

# --- Video capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected.")
    exit()

def update_video():
    global suspicious_counter, cal_h_min, cal_h_max, cal_v_min, cal_v_max

    success, frame = cap.read()
    if not success:
        root.after(10, update_video)
        return

    frame = cv2.flip(frame, 1)  # Flip like original gaze detector
    height, width, _ = frame.shape

    # --- Eye gaze detection on clean frame ---
    h_ratio, v_ratio = get_gaze_ratio(frame)

    # --- Automatic calibration ---
    elapsed = (datetime.now() - calibration_start_time).total_seconds()
    is_calibrating = elapsed < CALIBRATION_DURATION

    if is_calibrating:
        suspicious_counter = 0
        gaze_history_x.clear()
        gaze_history_y.clear()
        cal_h_min = min(cal_h_min, h_ratio)
        cal_h_max = max(cal_h_max, h_ratio)
        cal_v_min = min(cal_v_min, v_ratio)
        cal_v_max = max(cal_v_max, v_ratio)
    else:
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

        # --- Determine gaze direction ---
        text_h = "CENTER"
        text_v = "CENTER"
        if avg_x < 0.25: text_h = "RIGHT"
        elif avg_x > 0.75: text_h = "LEFT"
        if avg_y < 0.25: text_v = "DOWN"
        elif avg_y > 0.75: text_v = "UP"

        if text_h != "CENTER" or text_v != "CENTER":
            suspicious_counter += 1
        else:
            suspicious_counter = max(0, suspicious_counter - 2)

    # --- Run object detection on a copy of the frame ---
    frame_for_yolo = frame.copy()
    annotated_frame, cheating_detected, detected_objects = detect_objects(model, frame_for_yolo)

    # --- Overlay gaze info on annotated frame ---
    if not is_calibrating:
        is_suspicious = suspicious_counter > SUSPICIOUS_TIME_THRESHOLD
        color = (0, 255, 0)
        if is_suspicious:
            color = (0, 0, 255)
            cv2.putText(annotated_frame, "SUSPICIOUS ACTIVITY: LOOKING AWAY",
                        (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.rectangle(annotated_frame, (0, 0), (width, height), (0, 0, 255), 10)
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open("cheating_log.txt", "a") as f:
                f.write(f"[{timestamp}] SUSPICIOUS ACTIVITY DETECTED\n")
            if suspicious_counter % 50 == 0:
                filename = f"evidence_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Evidence saved: {filename}")
        elif suspicious_counter > 10:
            color = (0, 255, 255)
            cv2.putText(annotated_frame, "Warning: Focus...", (50, height - 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        cv2.putText(annotated_frame, f"{text_h} - {text_v}", (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    else:
        cv2.putText(annotated_frame, f"CALIBRATING... ({int(CALIBRATION_DURATION - elapsed)}s)",
                    (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    # --- Update GUI status ---
    all_cheating = cheating_detected or (suspicious_counter > SUSPICIOUS_TIME_THRESHOLD)
    if all_cheating:
        status_lbl.config(text="CHEATING DETECTED", fg="red")
    else:
        status_lbl.config(text="STATUS: CLEAN", fg="green")
    info_lbl.config(text=f"Visible: {', '.join(detected_objects)}")

    # --- Display frame in Tkinter ---
    rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    img = img.resize((800, 450))
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)

    # --- Schedule next update ---
    root.after(10, update_video)

# --- Start GUI loop ---
update_video()
root.mainloop()
cap.release()
