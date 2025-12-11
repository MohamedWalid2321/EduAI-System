# ObjectDetectionApp.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from ObjectDetection import load_model, detect_objects, cheating_classes

# --- Load YOLO model ---
model = load_model("D:\\Downloads\\GradPro\\AI\\Models\\objectDetectionYolo\\best.pt")

# --- GUI ---
root = tk.Tk()
root.title("AI Proctoring System - Object Detection")
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

def update_video():
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)  # Flip like gaze detector

        annotated_frame, cheating_detected, detected_objects = detect_objects(model, frame)

        # --- Update GUI ---
        if cheating_detected:
            status_lbl.config(text="CHEATING DETECTED", fg="red")
            info_lbl.config(text=f"Detected: {', '.join(detected_objects)}")
        else:
            status_lbl.config(text="STATUS: CLEAN", fg="green")
            info_lbl.config(text=f"Visible: {', '.join(detected_objects)}")

        # --- Display frame ---
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        img = img.resize((800, 450))
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)

    root.after(10, update_video)

update_video()
root.mainloop()
cap.release()
