"""
Bu sınıf;
    - fotoğrafları veri tabanında alır
    - OpenCV ile yüzlerini bulur ve keser
    - Kesilen yüzleri FaceNet modele uygun hale getir.

Son durumda kişilere ait yüzleri sözlük yapsında döndürür.
"""
import glob
import os
import cv2
import numpy as np
from keras.models import load_model


class Preprocess:
    def __init__(self, database_path,folders):
        self.path = database_path
        self.folders = folders
        self.model =  load_model('./Model/facenet_keras.h5')
        print("[Log] Preprocess Nesnesi Oluşturuldu.",flush=True)

    def load_images(self):    
        database = {}
        os.chdir(self.path)
        for i in self.folders:
            for file in glob.glob(i+"/*.jpg"):
                img = cv2.imread(file)
                #self.show(img)
                faces = self.getFace(img)            
                if faces != None:
                    for face in faces:
                        x = cv2.resize(face, (160, 160))
                        database[i] = self.embedding(x)
        print("[Log] Database Oluşturuldu.",flush=True)
        return database

    def embedding(self,img):
        img = img[...,::-1]
        img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
        img = np.array([img])
        embedding = self.model.predict_on_batch(img)
        return embedding[0]

    def getFace(self, img):
        f = []
        face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            face_image = img[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]    
            f.append(face_image)
        return f

    def show(self,img):
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
#p = Preprocess(database_path="./Database/",folders=["Sherlock","John"])
#dic = p.load_images()
#print(dic)