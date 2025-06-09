#  Face Recognition System (LBPH + OpenCV)

This is a real-time face recognition system developed using **OpenCV** and the **LBPH (Local Binary Patterns Histograms)** algorithm. It detects and recognizes faces using a webcam and Haar cascades.

---

## Overview

- **Project Type**: Self Project  
- **Technologies Used**: Python, OpenCV, LBPH, Haar Cascades  
- **Project Duration**: Summer 2024  
- **Purpose**: To detect and recognize faces from webcam input using pre-trained facial data.

---



## How It Works

1. Face Detection
Technique Used: Haar Cascade Classifier

Description: Converts input images to grayscale and uses OpenCV’s pre-trained Haar Cascade model to detect faces. This method identifies features like eyes, nose, and mouth patterns to locate a face in the image.

2. Face Recognition
Technique Used: Local Binary Patterns Histograms (LBPH)

Description: Recognizes the faces detected in the previous step by comparing them with known faces in the training dataset. LBPH is robust to lighting variations and is effective for real-time face recognition tasks.

3. Training Dataset - File "A"
Content: Contains labeled images of different individuals. Each label corresponds to a unique person (e.g., "A", "B", etc.).

Usage: These labeled face images are used to train the LBPH recognizer so that it can identify individuals based on the features extracted during training.



