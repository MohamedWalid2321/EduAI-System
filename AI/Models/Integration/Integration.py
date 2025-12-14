# integration.py
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import cv2
from datetime import datetime
from collections import deque
import numpy as np
import threading
import sys
import os

# --- Import logic from other modules ---
from ObjectDetection import load_model, detect_objects
from GazeTest import get_gaze_ratio  # assumes GazeTest.py contains the function

# Add Face Recognition Service path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Face_Recognition_Service"))
from face_recognition import FaceRecognitionService

# --- Configuration ---
SMOOTHING_BUFFER_SIZE = 3
SUSPICIOUS_TIME_THRESHOLD = 45
CALIBRATION_DURATION = 15  # seconds
FACE_VERIFICATION_INTERVAL = 30  # frames (~1 second)

# --- Load YOLO model ---
model = load_model("C:\\Users\\hagar\\OneDrive\\Documents\\Prev.IG_Current CE\\Grad-Project\\EduAI-System\\AI\\Models\\objectDetectionYolo\\best.pt")

# --- Initialize Face Recognition Service ---
reference_images_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reference_images")
face_service = FaceRecognitionService(
    reference_images_path=reference_images_path,
    model_name="VGG-Face",
    detector_backend="ssd"
)

# --- Face Detection for UI (Haar Cascade - fast) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Face Recognition State ---
face_verification_result = {
    "verified": False,
    "status": "Not Verified",
    "color": (0, 165, 255),  # Orange
    "distance": 0,
    "last_check": None
}
face_verification_lock = threading.Lock()
face_verification_running = False
registered_student_id = None  # Will be set when exam starts

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
root.geometry("1000x750")

title_lbl = tk.Label(root, text="Exam Session In Progress", font=("Helvetica", 24, "bold"))
title_lbl.pack(pady=10)

video_frame = tk.Label(root)
video_frame.pack()

# --- Face Recognition Status Panel ---
face_panel = tk.Frame(root, bg="#e0e0e0", pady=5)
face_panel.pack(fill="x")

face_status_lbl = tk.Label(face_panel, text="IDENTITY: Not Verified", font=("Arial", 14, "bold"), fg="orange", bg="#e0e0e0")
face_status_lbl.pack(side="left", padx=20)

student_id_lbl = tk.Label(face_panel, text="Student: None", font=("Arial", 12), bg="#e0e0e0")
student_id_lbl.pack(side="left", padx=20)

# --- Main Status Panel ---
status_panel = tk.Frame(root, bg="#f0f0f0", pady=20)
status_panel.pack(fill="x", side="bottom")

status_lbl = tk.Label(status_panel, text="STATUS: CLEAN", font=("Arial", 20, "bold"), fg="green", bg="#f0f0f0")
status_lbl.pack()

info_lbl = tk.Label(status_panel, text="No anomalies detected.", font=("Arial", 12), bg="#f0f0f0")
info_lbl.pack()

btn_quit = ttk.Button(root, text="End Exam", command=root.destroy)
btn_quit.place(x=900, y=20)


def set_student_id():
    """Prompt for student ID at start of exam"""
    global registered_student_id
    student_id = simpledialog.askstring("Student Verification", "Enter Student ID:", parent=root)
    if student_id:
        registered_student_id = student_id
        student_id_lbl.config(text=f"Student: {student_id}")
        # Check if reference image exists
        ref_path = os.path.join(reference_images_path, f"{student_id}.jpg")
        if os.path.exists(ref_path):
            print(f"Reference image found for: {student_id}")
        else:
            print(f"WARNING: No reference image found for {student_id}")
            face_status_lbl.config(text="IDENTITY: No Reference Image", fg="red")


def verify_face_background(frame_to_verify):
    """Run face verification in background thread"""
    global face_verification_running
    
    if registered_student_id is None:
        face_verification_running = False
        return
    
    try:
        result = face_service.verify_face_from_frame(frame_to_verify, registered_student_id)
        
        with face_verification_lock:
            face_verification_result["distance"] = result.get("distance", 0)
            face_verification_result["last_check"] = datetime.now()
            
            if result.get("verified"):
                face_verification_result["verified"] = True
                face_verification_result["status"] = "VERIFIED"
                face_verification_result["color"] = (0, 255, 0)  # Green
            else:
                face_verification_result["verified"] = False
                error = result.get("error", "")
                if "could not be detected" in str(error).lower():
                    face_verification_result["status"] = "No Face"
                    face_verification_result["color"] = (0, 165, 255)  # Orange
                else:
                    face_verification_result["status"] = "WRONG PERSON"
                    face_verification_result["color"] = (0, 0, 255)  # Red
                    # Log identity violation
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    with open("cheating_log.txt", "a") as f:
                        f.write(f"[{timestamp}] IDENTITY VIOLATION: Face does not match {registered_student_id}\n")
    except Exception as e:
        with face_verification_lock:
            face_verification_result["status"] = "Error"
            face_verification_result["color"] = (0, 0, 255)
    finally:
        face_verification_running = False

# --- Video capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected.")
    exit()

frame_count = 0  # For face verification interval

def update_video():
    global suspicious_counter, cal_h_min, cal_h_max, cal_v_min, cal_v_max
    global face_verification_running, frame_count

    success, frame = cap.read()
    if not success:
        root.after(10, update_video)
        return

    frame = cv2.flip(frame, 1)  # Flip like original gaze detector
    height, width, _ = frame.shape
    frame_count += 1

    # --- Face Verification (Background Thread) ---
    if frame_count % FACE_VERIFICATION_INTERVAL == 0 and not face_verification_running and registered_student_id:
        face_verification_running = True
        thread = threading.Thread(target=verify_face_background, args=(frame.copy(),))
        thread.daemon = True
        thread.start()

    # --- Detect faces for UI bounding box (Haar Cascade - fast) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

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

    # --- Get Face Verification Status ---
    face_verified = False
    face_status = "Not Verified"
    face_color_bgr = (0, 165, 255)  # Default orange
    
    with face_verification_lock:
        face_verified = face_verification_result["verified"]
        face_status = face_verification_result["status"]
        face_color_bgr = face_verification_result["color"]

    # --- Draw Face Bounding Box ---
    for (x, y, w, h) in faces:
        # Use verification status color for the box
        box_color = face_color_bgr
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
        # Add label above the box
        label = f"{face_status}"
        cv2.rectangle(annotated_frame, (x, y - 25), (x + len(label) * 10 + 10, y), box_color, -1)
        cv2.putText(annotated_frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Draw Face Verification Status on frame ---
    face_indicator_color = face_color_bgr
    cv2.rectangle(annotated_frame, (width - 200, 10), (width - 10, 60), (50, 50, 50), -1)
    cv2.putText(annotated_frame, f"ID: {face_status}", (width - 190, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_indicator_color, 2)

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
    # Check for identity violation (wrong person or no face for too long)
    identity_violation = (registered_student_id and face_status == "WRONG PERSON")
    
    all_cheating = cheating_detected or (suspicious_counter > SUSPICIOUS_TIME_THRESHOLD) or identity_violation
    
    if identity_violation:
        status_lbl.config(text="IDENTITY VIOLATION!", fg="red")
    elif all_cheating:
        status_lbl.config(text="CHEATING DETECTED", fg="red")
    else:
        status_lbl.config(text="STATUS: CLEAN", fg="green")
    
    # Update face status label
    if face_verified:
        face_status_lbl.config(text=f"IDENTITY: Verified ({registered_student_id})", fg="green")
    elif face_status == "WRONG PERSON":
        face_status_lbl.config(text="IDENTITY: WRONG PERSON!", fg="red")
    elif face_status == "No Face":
        face_status_lbl.config(text="IDENTITY: No Face Detected", fg="orange")
    else:
        face_status_lbl.config(text=f"IDENTITY: {face_status}", fg="orange")
    
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
# Ask for student ID before starting
root.after(100, set_student_id)  # Prompt for student ID after GUI loads
update_video()
root.mainloop()
cap.release()
