import cv2 as cv

haar_cascade = cv.CascadeClassifier("haarcascades.xml")

# Define the list of people (labels)
people = ['A']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# Open the camera
camera = cv.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture images from the camera
for i in range(1):
    ret, image = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        continue

    image_path = 'opencv' + str(i) + '.png'
    cv.imwrite(image_path, image)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)

        if 0 <= label < len(people):
            print(f'Label = {people[label]} with a confidence of {confidence}')

            cv.putText(image, str(people[label]), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

            if confidence >= 80:
                print("Please wait until the owner opens the door and door bell rings")
            else:
                print(f"Welcome to home {people[label]}")
        else:
            print("Unknown person. Please wait until owner opens the door.")

    cv.imshow('Captured Image', image)
    cv.waitKey(1000)

camera.release()
cv.destroyAllWindows()
