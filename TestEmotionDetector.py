import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import locale

# Ensure UTF-8 encoding is used for output
sys.stdout.reconfigure(encoding='utf-8')

# Load pre-trained model for emotion detection
try:
    model_path = r'D:\cv\archive (4)\emotion_model_full.h5'  # Update this path if necessary
    emotion_model = load_model(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please check the file path.")
    exit()

# Emotion labels corresponding to the model output
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected")
    else:
        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop the face from the image
            face_region = frame[y:y+h, x:x+w]
            face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Resize and preprocess image for model input
            face_resized = cv2.resize(face_region_gray, (48, 48))
            face_resized = face_resized.astype('float32') / 255  # Normalize the image
            face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
            face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

            # Predict the emotion
            try:
                emotion_prediction = emotion_model.predict(face_resized)
                max_index = np.argmax(emotion_prediction[0])
                predicted_emotion = emotion_labels[max_index]

                # Display detected emotion
                print(f"Detected emotion: {predicted_emotion}")

                # Draw rectangle and label on face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_emotion, 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error during emotion prediction: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
