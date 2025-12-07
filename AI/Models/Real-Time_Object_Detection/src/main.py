import cv2 as cv
from face_detection import detect_faces, get_landmarks

capture=cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open Camera")
    exit()

while True:
    ret, frame=capture.read() #capture frame-by-frame

    # if frame is read correctly ret is True
    if not ret:
        print("Cannot receive frames. Exiting...")
        break
    #frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    print(frame.dtype, frame.shape)
    rectFaces=detect_faces(frame) #list of faces

    for rect in rectFaces:
        land=get_landmarks(frame,rect)
        for (x,y) in land:
            cv.circle(frame,(x,y),2,(0,255,0),-1)

    cv.imshow('frame',frame)
    if cv.waitKey(1)==ord('q'):
        break

cv.release()
cv.destroyAllWindows()    
