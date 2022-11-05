import cv2
import numpy as np

confidence_limit = 0.8

capture = cv2.VideoCapture(1)
cv2.namedWindow("Face detector", cv2.WINDOW_AUTOSIZE)

net = cv2.dnn.readNetFromCaffe("./data/deploy.prototxt.txt", "./data/res10_300x300_ssd_iter_140000.caffemodel")

while cv2.waitKey(1) & 0xFF != ord('q'):
    _, frame = capture.read()
    frame_flip = cv2.flip(frame, 1)

    h, w = frame_flip.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_flip, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_limit:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")

        text = f"{confidence * 100:.2f}%"
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.rectangle(frame_flip, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame_flip, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face detector", frame_flip)

capture.release()
cv2.destroyWindow("Face detector")
