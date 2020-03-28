import cv2
import numpy as np
import os
cap = cv2.VideoCapture(-1)
i=0
no_samples= 500
folder_name = 'data/' +input()+'/'
start = False
while True:
    ret, frame = cap.read()
    # key = cv2.waitKey(10)
    if ret!=True or i==no_samples:
        break
    frame=cv2.rectangle(frame,(0,100),(280,400),(255,255,0),4)
    cv2.imshow('full', frame)
    if start:
        x, y, w, h = 0, 100, 280, 300
        img = frame[y:y + h, x:x + w]
        cv2.imshow('frames',frame)
        cv2.imwrite(folder_name+'frame{}.jpg'.format(i), img)
        i += 1
    key =cv2.waitKey(30) & 0xff
    if key == ord('p'):
        start = True
    if key == ord('s'):
        start = False
    if key == 27:
        break
print(len(os.listdir(folder_name)))
cap.release()
cv2.destroyAllWindows()