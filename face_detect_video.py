import cv2 as cv
import numpy as np

video = cv.VideoCapture('images/video1.mp4')

haar_face = cv.CascadeClassifier('haar_face.xml')

def resized(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width,height)
    return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)

while True:
    isTrue, frame = video.read()
    frame = resized(frame, scale=0.5)
    if not isTrue:
        break
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face_rect = haar_face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    n = len(face_rect)
    text = f'Faces : {n}'
    cv.putText(frame,text,(10,30),cv.FONT_HERSHEY_SIMPLEX,1,255,3)

    for x,y,w,h in face_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    cv.imshow('video',frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

    
video.release()
cv.destroyAllWindows()

