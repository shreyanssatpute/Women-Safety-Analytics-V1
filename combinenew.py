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
cv2.namedWindow("Age, Gender, and Violence Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Age, Gender, and Violence Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Weight bias to prioritize female detection
female_bias_weight = 1.2  # Adjust this weight to prioritize females more

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

        # Predict gender with weight bias
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()

        # Apply the bias to female prediction
        female_pred = gender_preds[0][1] * female_bias_weight
        male_pred = gender_preds[0][0]

        # Final gender decision based on biased weights
        if female_pred > male_pred:
            gender = 'Female'
            female_count += 1
        else:
            gender = 'Male'
            male_count += 1

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label = f'{gender}, Age: {age}'
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
    violence_confidence = prediction if prediction > 0.5 else 1 - prediction

    # Detect whether violence is predicted
    violence_label = 'violence' if prediction <= 0.5 else 'non-violence'

    # Display the violence prediction on the frame
    cv2.putText(frame, f"Violence: {violence_label} ({violence_confidence:.2f})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Alert conditions
    # Condition 1: When more than 3 males are surrounding a single female
    alert_percentage = 0
    if female_count > 0 and male_count > 3:
        ratio = male_count / female_count
        if ratio > 3:
            alert_percentage = min(100, (ratio / 3) * 50)  # Increase alert level based on male-to-female ratio
            alert_msg = "ALERT: Female surrounded by more than 3 males"
            cv2.putText(frame, alert_msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Condition 2: If violence is detected along with condition 1
    if violence_label == 'violence' and female_count > 0 and male_count > 3:
        alert_percentage += 50  # Increase alert level due to violence
        alert_msg_violence = "ALERT: Female surrounded by males with violence!"
        cv2.putText(frame, alert_msg_violence, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the alert percentage
    cv2.putText(frame, f"Alert Level: {alert_percentage:.2f}%", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with predictions
    cv2.imshow("Age, Gender, and Violence Detection", frame)

    # Press 'q' to quit the real-time prediction
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()