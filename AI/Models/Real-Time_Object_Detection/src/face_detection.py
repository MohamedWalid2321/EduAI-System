import dlib
import cv2 as cv
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\Downloads\GradPro\AI\Models\Real-Time Object Detection & Tracking in Video Streams\dlib_Predicator\shape_predictor_68_face_landmarks.dat")

def detect_faces(frame):
    # return list of rectangles
    rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB).copy(order='C')
    rgb = np.ascontiguousarray(rgb)

    print("RGB:", rgb.dtype, rgb.shape)
    print("CONTIGUOUS?", rgb.flags['C_CONTIGUOUS'])
    rect=detector(rgb)
    return rect


def get_landmarks(frame, rect):
    # return 68 landmark points as list of (x, y)
    rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    shape=predictor(rgb,rect)

    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return landmarks