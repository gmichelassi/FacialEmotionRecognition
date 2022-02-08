import cv2
import numpy as np

from neural_network.utils.load_dataset import label_to_text
from tensorflow.keras.models import load_model

MODEL = load_model('./model.h5')
VIDEO = cv2.VideoCapture(0)
FACE_CASCADE = cv2.CascadeClassifier('./cascades/haar_frontalface_default.xml')
RED = (0, 0, 255)
YELLOW = (0, 120, 255)


def capture():
    stop_recording = False
    while not stop_recording:
        _, frame = VIDEO.read()

        find_face(frame)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_recording = True

    VIDEO.release()
    cv2.destroyAllWindows()


def find_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print(MODEL.predict(frame))

    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for x, y, width, height in faces:
        cropped_image_around_face = frame[y:y + height, x:x + width]
        resized_img = cv2.resize(cropped_image_around_face, (48, 48), interpolation=cv2.INTER_AREA)
        images_list = np.asarray([np.array(resized_img)])
        predictions = MODEL.predict(images_list)[0]
        predicted_index = np.argmax(predictions)
        label = label_to_text[predicted_index]

        print(label)

        draw_rectangle(frame, x, y, width, height)
        write_text(frame, label, x, y)


def draw_rectangle(frame, x, y, width, height):
    cv2.rectangle(
        img=frame,
        pt1=(x, y),
        pt2=(x + width, y + height),
        color=RED,
        thickness=2
    )


def write_text(frame, text, x, y):
    cv2.putText(
        img=frame,
        text=text,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=3,
        color=YELLOW,
        thickness=2
    )