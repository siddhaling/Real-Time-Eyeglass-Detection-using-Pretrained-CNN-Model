import cv2
import numpy as np


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

model = load_model('softmax.h5', custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

cam = cv2.VideoCapture(1)


ret, frame1 = cam.read()
frame1_c = frame1[110:540,460:865]
ret, frame2 = cam.read()
frame2_c = frame2[110:540,460:865]


flag = 0
s_class = -1

while True:
    diff = cv2.absdiff(frame1_c,frame2_c)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thres = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thres, None, iterations=3)
    contours ,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    faces = cv2.cvtColor(frame1_c, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(faces, 1.1, 4)
    frame = []
    for (x,y,w,z) in faces:
        cv2.rectangle(frame1_c, (x-15,y-40), (x+w+15,y+z+40), (255,0,0), 3)
        frame1_c = frame1_c[y-40:y+z+40, x-15:x+w+15]

    img = np.array(frame1_c)
    if not(img.size == 0):
        img = tf.image.resize(img,[299,299])
        img = tf.cast(img/255. ,tf.float32)
        img = np.expand_dims(img, axis=0)
        pre = model.predict(img)
        f_class = np.argmax(pre)
    else:
        s_class = -1

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 10000:
            if flag == 20:
                if f_class == 0:
                    s_class = "Present"
                elif f_class == 1:
                    s_class = "Absent"


            else:
                flag += 1
        else:
            flag = 0




    #cv2.drawContours(frame1_c, contours, -1, (0,255,0), 2)

    cv2.rectangle(frame1, (450, 100), (875, 550), (105, 5, 99), 2)
    frame_s = cv2.flip(frame1,1)
    cv2.putText(frame_s, "Class: {}".format(s_class), (10,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),3)
    cv2.imshow("frame", frame_s)


    frame1 = frame2
    frame1_c = frame2_c
    ret, frame2 = cam.read()
    frame2_c = frame2[110:540,460:865]



    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
