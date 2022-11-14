import os

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential

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


# ----------------------------------------------Extraction-----------------------------------------------------------

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


np.load('C:\\Users\\Gourav Kumar\\OneDrive\\Desktop\\Project work\\Motion Detection\\Body_Pose_Detection\\New.npy')


# # 3. Data Collection

# In[7]:


# Path for exported data, numpy arrays
DataPath = os.path.join('NewData')

# Actions that we try to detect
actions = np.array(['Hello', 'Namaste', 'Danger', 'Walking', 'Angry',
                   'Drowsy/Sleepy', 'Suspicious', 'Laughing', 'Sad', 'Victory'])

# Thirty videos worth of data
numOfSequences = 30

# Videos are going to be 30 frames in length
sequenceLength = 30

# Folder start
startFolder = 30


# In[8]:


for action in actions:
    for sequence in range(numOfSequences):
        try:
            os.makedirs(os.path.join(DataPath, action, str(sequence)))
        except:
            pass


##sequences, labels = [], []
# for action in actions:
#     for sequence in range(numOfSequences):
#         window = []
#         for frame_num in range(sequenceLength):
#             res = np.load(os.path.join(DataPath, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])


# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.models import Sequential

# In[25]:


# log_dir = os.path.join('LogsDirectory')
# tb_callback = TensorBoard(log_dir=log_dir)


# In[26]:


model = Sequential()
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.load_weights(
    'C:\\Users\\Gourav Kumar\\OneDrive\\Desktop\\Project work\\Motion Detection\\Body_Pose_Detection\\action_detection.h5')


colors = [(255, 255, 31), (117, 245, 16), (255, 128, 0), (28, 255, 248), (225, 28, 28),
          (0, 204, 204), (204, 0, 204), (16, 117, 245),  (0, 0, 204), (255, 255, 51)]


def probabilityVisualize(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


# In[32]:


# plt.figure(figsize=(18,18))
# plt.imshow(probabilityVisualize(res, actions, image, colors))


# In[30]:


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.7

cap = cv2.VideoCapture(0)
# Set mediapipe model
with holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

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
        sequence = sequence[:30]
#         sequence.append(keypoints)
#         sequence = sequence[-30:]

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
            image = probabilityVisualize(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Window', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[22]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:
