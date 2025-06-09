import os
import cv2 as cv
import numpy as np

people = ['A']
DIR = "A"  
haar_cascade_path = "haarcascades.xml"


if not os.path.exists(haar_cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found: {haar_cascade_path}")

haar_cascade = cv.CascadeClassifier(haar_cascade_path)

features = []
labels = []

def create_train():
    if not os.path.isdir(DIR):
        raise NotADirectoryError(f"Directory does not exist: {DIR}")

    for person in people:
        path = os.path.join(DIR, person)
        if not os.path.isdir(path):
            print(f"Person directory does not exist: {path}")
            continue
        
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Could not read image: {img_path}")
                continue 

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
