from  preprocess import Preprocess
from keras.models import load_model
import numpy as np
import cv2
import config

class FaceRecognition:
    def __init__(self):
        """initialize dataset and load model"""
        self.model = load_model(config.model_path)
        print("[Log] Pretrained model was loaded.")
        
        self.preprocess = Preprocess(database_path=config.database_path)
        print("[Log] Preprocess object was created.")
        
        self.database = self.init_database()
        
        
    def init_database(self):
        """ initilize face database"""
        return self.preprocess.load_images()
        print("[Log] Database initialized.",flush=True)

    
    def recognize_faces_in_video(self, videopath):
        
        vc = cv2.VideoCapture(videopath)
        while vc.isOpened():
            _, frame = vc.read()
            height, width, channels = frame.shape
            
            faces, coordinates = self.preprocess.getFace(frame)    
            if faces == None:
                continue
            # Loop through all the faces
            for i, face in enumerate(faces):
                 
                embeded_face = self.preprocess.embedding(face)
                name, similarity =  self.findFaceInDB(embeded_face)
                #print(name,"-",similarity)
                if similarity <= config.threshold:
                    x1, y1, x2, y2 = coordinates[i]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.putText(frame,name, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    
            key = cv2.waitKey(100)
            cv2.imshow("Face Recognizer", frame)
            if key == 27: # exit on ESC
                break
        
        vc.release()
        cv2.destroyAllWindows()

    def findFaceInDB(self, embedded_face):
        """ This method finds person in person database."""    
        who, similarity = None, None
        min_similarity = 200

        persons = list(self.database.keys())
        
        for person in persons:
            for vec in self.database[person]:
                sim = self.preprocess.euclid_distance(embedded_face,vec)
                if sim < min_similarity:
                    min_similarity = sim
                    who = person
                    
            
        return who, min_similarity


                


if __name__ == "__main__":    
    FR = FaceRecognition()
    FR.recognize_faces_in_video("/home/mp/Desktop/github/Face-Recognition-with-FaceNet-and-OpenCV/test-videos/test.mp4")



