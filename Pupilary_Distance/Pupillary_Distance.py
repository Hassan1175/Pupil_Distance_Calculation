#print("Hello word .. .")
import numpy as np
import math, datetime
import dlib, cv2
import os, sys



#This is the Caffe based ResNet+SSD model for face detection to be used in OpenCV
FACE_MODEL_PATH = "{base_path}/deps/res10_300x300_ssd_iter_140000.caffemodel".format(base_path=BASE_PATH)
FACE_PROTO_PATH =  "{base_path}/deps/deploy.prototxt.txt".format(base_path=BASE_PATH)


face_detector_cv = cv2.dnn.readNetFromCaffe(FACE_PROTO_PATH, FACE_MODEL_PATH)





#That method has been implemented in order find out that either lightening condition 
#is sufficient or not . . . . That method also has not been used in the final version of code . . .

def isLightAdequate(y):
    NUM_BINS = 10;
    total_pixels = y.shape[0] * y.shape[1];
    hist = cv2.calcHist([y],[0],None,[256],[0,256])
    hist = [i[0] for i in hist]
    
    dark_values = hist[0:NUM_BINS];
    light_values = hist[-NUM_BINS:]

    light = round(sum(light_values) / total_pixels * 100.0, 3)
    dark = round(sum(dark_values) / total_pixels * 100.0, 3)

    if abs(dark - light) >= 5:
        return False, "Too dark/Too bright | "
    else:
        return True, ""


#Apply some pre-processing to the image before detecting face in order to adjust the brightness of the image
#That method is not more in use in the code, but still has been developed, just for the testing purpose . .. 
def preProcess(bgr_image):
    print('Applying CLAHE')
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    adequateLight, lightMessage = isLightAdequate(ycrcb[:,:,0])
    ycrcb[:,:,0] = CLAHE.apply(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), adequateLight, lightMessage




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
