import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
import time
import tempfile
from PIL import Image


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.m4v'


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
    # for num, prob in enumerate(res):
    #     cv2.rectangle(output_frame, (0, 60+num*40),
    #                   (int(prob*100), 90+num*40), -1)
    #     cv2.putText(output_frame, actions[num], (0, 85+num*40),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


# 1. New detection variables
sequence = []
sentence = ""
threshold = 0.7

FRAME_WINDOW = st.image([])

st.set_option('deprecation.showfileUploaderEncoding', False)

use_webcame = st.sidebar.button('Use WebCame')
close_webcame = st.sidebar.button('Close Webcame')
record = st.sidebar.checkbox("Record Video")

if record:
    st.checkbox("Recording", value=True)

st.markdown(
    """
    <style>
    [data-testid="stSiderbar"][aria-expanded="true"] > div:first-child{
        width : 350 px
    }
    [data-testid="stSiderbar"][aria-expanded="false"] > div:first-child{
        width : 350 px
        margin-left: -350px
    }
    </style>
    """,

    unsafe_allow_html=True,
)

# st.sidebar.markdown('----')
# detection_confidence = st.sidebar.slider(
#     'Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
# tracking_confidence = st.sidebar.slider(
#     'Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

st.sidebar.markdown('----')

st.markdown("## Output")

## WE GET OUR VIDEO INPUT ##
stframe = st.empty()
video_file_buffer = st.sidebar.file_uploader(
    "Upload a Video", type=["mp4", "mov", "avi", "asf", "m4v"])
tffile = tempfile.NamedTemporaryFile(delete=False)

if not video_file_buffer:
    if use_webcame:
        vid = cv2.VideoCapture(0)

    else:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tffile.name = DEMO_VIDEO

else:
    tffile.write(video_file_buffer.read())
    vid = cv2.VideoCapture(tffile.name)

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(vid.get(cv2.CAP_PROP_FPS))

## RECORDING PART ##
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('output1.m4v', codec, fps_input, (width, height))

st.sidebar.text('Input Video')
st.sidebar.video(tffile.name)

fps = 0
i = 0

drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

kpi1 = st.markdown("Please excersie")

with kpi1:
    st.markdown("**Output**")
    kpi1_text1 = st.markdown("0")


st.markdown("<hr/>", unsafe_allow_html=True)

# Set mediapipe model
with holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    prevTime = 0
    while (vid):

        # Read feed
        ret, frame = vid.read()

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
                        sentence = (actions[np.argmax(res)])
                else:
                    sentence = (actions[np.argmax(res)])

#             # Visualise probabilities
            image = probabilityVisualize(res, actions, image)

        # FPS COUNTER LOGIC
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        kpi1_text1.write(
            f"<h3>{str(sentence)}</h3>", unsafe_allow_html=True)

        kpi1_text1.write(
            f"<h3>{str(sentence)}</h3>", unsafe_allow_html=True)

        # Show to screen
        stframe.image(image, channels='BGR', use_column_width=True)
        # FRAME_WINDOW.image(frame)

        # Break gracefully
        if close_webcame:

            vid.release()
            cv2.destroyAllWindows()
            break
