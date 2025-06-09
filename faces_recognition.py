import cv2 as cv

# Load the pre-trained Haar cascade for face detection
haar_cascade = cv.CascadeClassifier(r"C:\Users\tyesw\OneDrive\Desktop\coding\New folder\haarcascades.xml")

# Define the list of people (labels)
people = ['A']

# Load the pre-trained LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\tyesw\OneDrive\Desktop\coding\New folder\face_trained.yml')

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
    
    # Save the captured image
    image_path = 'opencv' + str(i) + '.png'
    cv.imwrite(image_path, image)

    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Process each detected face
    for (x, y, w, h) in faces_rect:
        # Extract the region of interest (ROI) which is the face area
        faces_roi = gray[y:y+h, x:x+w]

        # Use the face recognizer to predict the person label and confidence
        label, confidence = face_recognizer.predict(faces_roi)

        # Check if the predicted label is within the valid range
        if 0 <= label < len(people):
            # Print the predicted label and confidence
            print(f'Label = {people[label]} with a confidence of {confidence}')

            # Draw text and rectangle around the detected face
            cv.putText(image, str(people[label]), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

            # Decision based on confidence level
            if confidence >= 80:
                print("Please wait until the owner opens the door and door bell rings")
            else:
                print("Welcome to home {people[label]}")
        else:
            print("Unknown person . please wait until owner open the door . door bell rings .")

    # Display the image with detected faces
    cv.imshow('Captured Image', image)

    # Wait for a short period to see the captured images
    cv.waitKey(1000)

# Release the camera and destroy all windows
camera.release()
cv.destroyAllWindows()
