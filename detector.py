import cv2
from urllib.request import urlopen
import numpy as np
url='http://192.168.43.1:8080/shot.jpg'



recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'dataSet'

#cam = cv2.VideoCapture(0)
while True:
    #ret, im =cam.read()
    imgResp = urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        tahminEdilenKisi, conf = recognizer.predict(gray[y:y + h, x:x + w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
        if(tahminEdilenKisi==1):
             tahminEdilenKisi= 'Emine Akgoz'
        elif (tahminEdilenKisi == 2):
            tahminEdilenKisi = 'İclal'
        elif (tahminEdilenKisi == 3):
            tahminEdilenKisi = 'Okan akgöz'
        else:
            tahminEdilenKisi= "Sistem Taniyamadi"
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        cv2.putText(img, str(tahminEdilenKisi), (x, y + h), fontFace, fontScale, fontColor)
        cv2.imshow('im',img)
        cv2.waitKey(10)









