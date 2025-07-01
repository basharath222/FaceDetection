import cv2 as cv
import numpy as np

img = cv.imread('images/group1.jpg')
img = cv.resize(img,(500,500))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

n = len(face_rect)
text = f'Faces : {n}'
cv.putText(img,text,(10,30),cv.FONT_HERSHEY_SIMPLEX,1,255,3)

for x,y,w,h in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('img',img)

cv.waitKey(0)