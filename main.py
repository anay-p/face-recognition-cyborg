import tensorflow as tf
import cv2
import numpy as np
from keras_vggface import utils

face_conf_limit = 0.8
person_conf_limit = 0.6
with open("list.txt") as f:
    list_str = f.read()
    person_list = list_str.split(",")

capture = cv2.VideoCapture(0)
cv2.namedWindow("Face recognizer", cv2.WINDOW_AUTOSIZE)

net = cv2.dnn.readNetFromCaffe("./data/deploy.prototxt.txt", "./data/res10_300x300_ssd_iter_140000.caffemodel")
model = tf.keras.models.load_model("model.h5")

while cv2.waitKey(1) & 0xFF != ord('q'):
    _, frame = capture.read()
    frame_flip = cv2.flip(frame, 1)

    h, w = frame_flip.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_flip, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        face_conf = detections[0, 0, i, 2]
        if face_conf < face_conf_limit:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0

        roi = frame_flip[start_y:end_y+1, start_x:end_x+1, :]
        resized_roi = cv2.resize(roi, (224, 224))
        x = resized_roi.reshape(1, 224, 224, 3)
        x = utils.preprocess_input(x.astype("float64"), version=2)

        person_conf_arr = model.predict(x, verbose=0)[0]
        person_id = person_conf_arr.argmax()
        person_conf = person_conf_arr[person_id]
        if person_conf >= person_conf_limit:
            text = f"{face_conf:.2f} | {person_conf:.2f} {person_list[person_id]}"
        else:
            text = f"{face_conf:.2f} | Unkown"

        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.rectangle(frame_flip, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame_flip, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face recognizer", frame_flip)

capture.release()
cv2.destroyWindow("Face recognizer")
