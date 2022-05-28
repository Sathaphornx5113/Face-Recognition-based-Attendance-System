import numpy as np
import cv2
import datetime
import time
import os
img_counter = 0
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

while True:
    ret, video = cap.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        dst = 6421 / w
        dst = '%.2f' %dst
        font = cv2.FONT_HERSHEY_SIMPLEX
        if(float(dst)>=50):
            
            img_name ="Attendance-{}.jpg".format(img_counter)
            cv2.imwrite(os.path.join("Attendance/")+ img_name,video)
            time.sleep(3)
            print("{} written!".format(img_name))
            img_counter += 1
            
    cv2.imshow('video', video)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
