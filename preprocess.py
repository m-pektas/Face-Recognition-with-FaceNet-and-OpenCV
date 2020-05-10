"""
This class can;
    - load photos from database
    - detect face and crop with opencv
    - create embedding vector from face.
"""
import os
import cv2
import numpy as np
from keras.models import load_model


class Preprocess:
    def __init__(self, database_path):
        self.path = database_path
        self.model =  load_model('model/facenet_keras.h5')
        self.face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

        print("[Log] Preprocess object was created.")

    def load_images(self):
        database = {}
        folders = os.listdir(self.path)
        for folder in folders:
            database[folder] = []
            files = os.listdir(os.path.join(self.path,folder))
            for file in files:
                filepath =  os.path.join(self.path,folder,file)
                img = cv2.imread(filepath)
                (faces, _) = self.getFace(img)            
                if faces != None:
                    for face in faces:
                        database[folder].append(self.embedding(face))
        print("[Log] Database was created.")
        return database
        

    def embedding(self,img):
        """ embed face with facenet model """
        img = img[...,::-1]
        img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
        img = np.array([img])
        embedding = self.model.predict_on_batch(img)
        return embedding[0]

    def getFace(self, img):
        face_list = []
        face_coor = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces)!=0:
            for (x, y, w, h) in faces:
                x1 = x
                y1 = y
                x2 = x+w
                y2 = y+h
                face_image = img[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]  
                face_image = cv2.resize(face_image, (160, 160))  
                face_list.append(face_image)
                face_coor.append((x1,y1,x2,y2))
            return (face_list,face_coor)
        else:
            return (None,None)

    def euclid_distance(self, input_embed, db_embed):
        """ calculate euclidan distance between two embeded vector """
        return np.linalg.norm(db_embed-input_embed)

    