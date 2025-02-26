import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the models for age and gender detection
face_proto = r"drive-download-20240828T144054Z-001/opencv_face_detector.pbtxt"
face_model = r"drive-download-20240828T144054Z-001/opencv_face_detector_uint8.pb"
age_proto = r"drive-download-20240828T144054Z-001/age_deploy.prototxt"
age_model = r"drive-download-20240828T144054Z-001/age_net.caffemodel"
gender_proto = r"drive-download-20240828T144054Z-001/gender_deploy.prototxt"
gender_model = r"drive-download-20240828T144054Z-001/gender_net.caffemodel"

# Mean values for age and gender detection
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load the face, age, and gender models
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Load the violence detection model
violence_model = tf.keras.models.load_model('violence_detection_model.h5')

# Define age and gender lists
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']
class_labels = ['violence', 'non-violence']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare image for face detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    # Detect faces
    face_net.setInput(blob)
    detections = face_net.forward()

    face_boxes = []
    male_count = 0
    female_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not face_boxes:
        print("No face detected")

    for face_box in face_boxes:
        face = frame[max(0, face_box[1]-15):min(face_box[3]+15, h-1),
                     max(0, face_box[0]-15):min(face_box[2]+15, w-1)]

        # Predict gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Increment gender counters
        if gender == 'Male':
            male_count += 1
        else:
            female_count += 1

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label = f'{gender}'
        cv2.putText(frame, label, (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the gender counts
    count_label = f'Male: {male_count}, Female: {female_count}'
    cv2.putText(frame, count_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Resize frame for violence detection
    resized_frame = cv2.resize(frame, (64, 64))
    img_array = img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction for violence detection
    prediction = violence_model.predict(img_array)[0][0]
    if prediction > 0.5:
        violence_label = class_labels[1]  # non-violence
        confidence = prediction
    else:
        violence_label = class_labels[0]  # violence
        confidence = 1 - prediction

    # Display the violence prediction on the frame
    cv2.putText(frame, f"Violence: {violence_label} ({confidence:.2f})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with predictions
    cv2.imshow("Age, Gender, and Violence Detection", frame)

    # Press 'q' to quit the real-time prediction
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
