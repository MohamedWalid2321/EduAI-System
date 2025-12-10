import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import winsound 

class ProctorApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("AI Proctoring System - Engineering Faculty")
        self.root.geometry("1000x700")
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        
        self.cheating_classes = [
            'Earphone', 
            'Mobile_phone', 
            'headset', 
            'smart_watch', 
            'sunglasses', 
            'cap'
        ]
        # GUI
        self.title_lbl = tk.Label(root, text="Exam Session In Progress", font=("Helvetica", 24, "bold"))
        self.title_lbl.pack(pady=10)

        # Video Frame
        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        # Status Panel
        self.status_panel = tk.Frame(root, bg="#f0f0f0", pady=20)
        self.status_panel.pack(fill="x", side="bottom")

        self.status_lbl = tk.Label(self.status_panel, text="STATUS: CLEAN", font=("Arial", 20, "bold"), fg="green", bg="#f0f0f0")
        self.status_lbl.pack()

        self.info_lbl = tk.Label(self.status_panel, text="No anomalies detected.", font=("Arial", 12), bg="#f0f0f0")
        self.info_lbl.pack()

        # Button to Quit
        self.btn_quit = ttk.Button(root, text="End Exam", command=self.close_app)
        self.btn_quit.place(x=900, y=20)

        # --- 3. Start Video ---
        self.cap = cv2.VideoCapture(0) # 0 is usually the default webcam
        self.running = True
        self.update_video()

    def update_video(self):
        if not self.running:
            return

        success, frame = self.cap.read()
        if success:
            results = self.model(frame, verbose=False, conf=0.5) 
            cheating_detected = False
            detected_objects = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    detected_objects.append(cls_name)
                    
                    if cls_name in self.cheating_classes:
                        cheating_detected = True

            annotated_frame = results[0].plot() 
            self.update_gui_status(cheating_detected, detected_objects)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
     
            img = img.resize((800, 450)) 
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def update_gui_status(self, is_cheating, objects):
        if is_cheating:
            self.status_lbl.config(text="CHEATING DETECTED", fg="red")
            self.info_lbl.config(text=f"Detected: {', '.join(objects)}")

        else:
            self.status_lbl.config(text="STATUS: CLEAN", fg="green")
            self.info_lbl.config(text=f"Visible: {', '.join(objects)}")

    def close_app(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProctorApp(root, "C:\\Users\\yoyo1\\OneDrive\\Desktop\\AIProctoring\\EduAI-System\\AI\\Models\\objectDetectionYolo\\best.pt") 
    root.mainloop()