import numpy as np
import cv2
import datetime
x = datetime.datetime.now() 
Day = str(x.strftime("%d-%m-%Y %H-%M-%S"))
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        dst = 6421 / w
        dst = '%.2f' %dst
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(dst), (x, y-10), font, 1, (0, 50, 250), 1, cv2.LINE_AA)
        #print(dst)
        if(float(dst)>=30):
            print(dst)
            img_name = str(Day)+".jpg"
            #cv2.imwrite("dataSet/JPEGImages/"+img_name, frame)         
            cv2.imwrite("Attendance/"+ img_name,img)
            print("{} written!".format(img_name))
            #print(Day)
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()