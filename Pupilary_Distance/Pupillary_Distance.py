#print("Hello word .. .")
import numpy as np
import math, datetime
import dlib, cv2
import os, sys



#This is the Caffe based ResNet+SSD model for face detection to be used in OpenCV
FACE_MODEL_PATH = "{base_path}/deps/res10_300x300_ssd_iter_140000.caffemodel".format(base_path=BASE_PATH)
FACE_PROTO_PATH =  "{base_path}/deps/deploy.prototxt.txt".format(base_path=BASE_PATH)


face_detector_cv = cv2.dnn.readNetFromCaffe(FACE_PROTO_PATH, FACE_MODEL_PATH)







def detectFaces(image):
    # print('Calling OpenCV DNN')
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector_cv.setInput(blob)
    detections = face_detector_cv.forward()

    confidence = detections[0, 0, 0, 2]

    retval = []

    for i in range(0, detections.shape[2]):
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            retval.append(dlib.rectangle(startX, startY, endX, endY))
    return retval
