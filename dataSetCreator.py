import numpy as np
import cv2
from urllib.request import urlopen




faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cam = cv2.VideoCapture(0)

url='http://192.168.43.1:8080/shot.jpg' #buraya ip kameranızın linkini yapıstıracaksınız

id = input('Kullanici id: ')
sampleNum = 0;
while (True):
    #ret, img = cam.read()
    imgResp = urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1;
        cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),5)
        cv2.waitKey(100);

    cv2.imshow('YüZ', img);
    cv2.waitKey(1);
    if (sampleNum > 25):
        break;
#cam.release()
cv2.destroyAllWindows()
