import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
import time


st.header("Virtual Excercise app")


##################################################################### Functions ###################################################################

holistics = mp.solutions.holistic  # To bring our holistic model
drawing = mp.solutions.drawing_utils  # Use fot drawing the utilities


def Detection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    img.flags.writeable = False                  # Image is no longer writeable
    results = model.process(img)                 # Make prediction
    img.flags.writeable = True                   # Image is now writeable
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return img, results


def DrawingCustomLandmarks(image, results):
    # Draw face connections
    drawing.draw_landmarks(image, results.face_landmarks, holistics.FACEMESH_TESSELATION,
                           drawing.DrawingSpec(
                               color=(80, 110, 10), thickness=1, circle_radius=1),
                           drawing.DrawingSpec(
                               color=(80, 256, 121), thickness=1, circle_radius=1)
                           )
    # Draw pose connections
    drawing.draw_landmarks(image, results.pose_landmarks, holistics.POSE_CONNECTIONS,
                           drawing.DrawingSpec(
                               color=(80, 22, 10), thickness=2, circle_radius=4),
                           drawing.DrawingSpec(
                               color=(80, 44, 121), thickness=2, circle_radius=2)
                           )
    # Draw left hand connections
    drawing.draw_landmarks(image, results.left_hand_landmarks, holistics.HAND_CONNECTIONS,
                           drawing.DrawingSpec(
                               color=(121, 22, 76), thickness=2, circle_radius=4),
                           drawing.DrawingSpec(
                               color=(121, 44, 250), thickness=2, circle_radius=2)
                           )
    # Draw right hand connections
    drawing.draw_landmarks(image, results.right_hand_landmarks, holistics.HAND_CONNECTIONS,
                           drawing.DrawingSpec(
                               color=(245, 117, 66), thickness=2, circle_radius=4),
                           drawing.DrawingSpec(
                               color=(245, 66, 230), thickness=2, circle_radius=2)
                           )

############################################################## 2. Extracting the Values ###################################################################


def ExtractingVals(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left, right])

############################################################## 3. Data Collection ###################################################################


# Path for exported data, numpy arrays
DataPath = os.path.join('ExerciseData')

# Actions that we try to detect
actions = np.array(['Push Ups', 'Lunges', 'Squats', 'Sit Ups', 'High Knees'])

# Thirty videos worth of data
numOfSequences = 30

# Videos are going to be 30 frames in length
sequenceLength = 30

# Folder start
startFolder = 30

############################################################## 4. Test in Real Time ###################################################################

model = load_model("Excercise.h5")

colors = [(255, 255, 31), (117, 245, 16), (255, 128, 0), (28, 255, 248), (225, 28, 28),
          (0, 204, 204), (204, 0, 204), (16, 117, 245),  (0, 0, 204), (255, 255, 51)]


def probabilityVisualize(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.7

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

# Set mediapipe model
with holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while (cap):

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = Detection(frame, holistic)
        print(results)

        # Draw landmarks
        DrawingCustomLandmarks(image, results)

        # 2. Prediction logic
        keypoints = ExtractingVals(results)
        sequence.insert(0, keypoints)
        print(sequence)
        sequence = sequence[:30]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

        # 3. Visualization logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

#             # Visualise probabilities
            image = probabilityVisualize(res, actions, image)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        FRAME_WINDOW.image(frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
