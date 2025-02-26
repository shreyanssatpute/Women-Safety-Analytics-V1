import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\rajla\Downloads\Violence Detetction\violence_detection_model.h5')

# Define class labels
class_labels = ['violence', 'non-violence']

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera, change index if needed

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Real-time video capture and prediction loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to match the input size expected by the model (e.g., 64x64)
    resized_frame = cv2.resize(frame, (64, 64))

    # Preprocess the frame: convert it to a numpy array, add batch dimension, and normalize
    img_array = img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Interpret the prediction
    if prediction > 0.5:
        label = class_labels[1]  # non-violence
        confidence = prediction
    else:
        label = class_labels[0]  # violence
        confidence = 1 - prediction

    # Display the label and confidence on the frame
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow("Real-Time Violence Detection", frame)

    # Press 'q' to quit the real-time prediction
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()