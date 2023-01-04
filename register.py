import cv2
import numpy as np
from keras_vggface import utils
import os
import sys

name = input("Enter name: ")
if os.path.isdir(f"images/{name}"):
    print("Already registered.")
    sys.exit()
os.mkdir(f"images/{name}")

confidence_limit = 0.8

capture = cv2.VideoCapture(0)
cv2.namedWindow("Face registerer", cv2.WINDOW_AUTOSIZE)

net = cv2.dnn.readNetFromCaffe("./data/deploy.prototxt.txt", "./data/res10_300x300_ssd_iter_140000.caffemodel")
registered = 0

while True:
    _, frame = capture.read()
    frame_flip = cv2.flip(frame, 1)
    frame_flip_copy = frame_flip.copy()

    h, w = frame_flip.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_flip, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    detected = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_limit:
            continue

        detected += 1

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")

        text = f"{confidence * 100:.2f}%"
        y = start_y - 10 if start_y - 10 > 10 else start_y + 20
        cv2.rectangle(frame_flip, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame_flip, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame_flip, f"Images taken: {registered}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face registerer", frame_flip)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if detected != 1:
            continue
        roi = frame_flip_copy[start_y:end_y+1, start_x:end_x+1, :]
        resized_roi = cv2.resize(roi, (224, 224))
        x = utils.preprocess_input(resized_roi.astype("float64"), version=2)
        cv2.imwrite(f"images/{name}/img{registered}.jpg", x.astype("uint8"))
        registered += 1
    elif key == ord('q'):
        if registered >= 10:
            break
        print("You must register at least 10 images.")

capture.release()
cv2.destroyWindow("Face registerer")
