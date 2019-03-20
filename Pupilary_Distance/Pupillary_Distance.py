#print("Hello word .. .")
import numpy as np
import math, datetime
import dlib, cv2
import os, sys



BASE_PATH = os.path.abspath(os.path.dirname(__file__))

CARD_DETECTOR_PATH = "{base_path}/deps/card_detector.svm".format(base_path=BASE_PATH)
FACE_DETECTOR_PATH = "{base_path}/deps/mmod_human_face_detector.dat".format(base_path=BASE_PATH)
EYE_DETECTOR_PATH = "{base_path}/deps/haarcascade_eye.xml".format(base_path=BASE_PATH)
LANDMARK_PREDICTOR_PATH = "{base_path}/deps/shape_predictor_68_face_landmarks.dat".format(base_path=BASE_PATH)
PUPIL_PREDICTOR_PATH = "{base_path}/deps/pupil_predictor.dat".format(base_path=BASE_PATH)

#This is the Caffe based ResNet+SSD model for face detection to be used in OpenCV
FACE_MODEL_PATH = "{base_path}/deps/res10_300x300_ssd_iter_140000.caffemodel".format(base_path=BASE_PATH)
FACE_PROTO_PATH =  "{base_path}/deps/deploy.prototxt.txt".format(base_path=BASE_PATH)



DLIB_CARD_PREDICTOR_PATH = "{base_path}/deps/card_predictor.dat".format(base_path=BASE_PATH)


dlib_card_predictor = dlib.shape_predictor(DLIB_CARD_PREDICTOR_PATH)
card_detector = dlib.simple_object_detector(CARD_DETECTOR_PATH);
face_detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_PATH) if USE_CNN else dlib.get_frontal_face_detector()
eye_detector = cv2.CascadeClassifier(EYE_DETECTOR_PATH);
landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)
pupil_predictor = dlib.shape_predictor(PUPIL_PREDICTOR_PATH)


face_detector_cv = cv2.dnn.readNetFromCaffe(FACE_PROTO_PATH, FACE_MODEL_PATH)




#Determine if a PD value is in valid range
def isValidPD(PD):
    PD = round(PD)
    # print(PD, MINIMUM_PD, MAXIMUM_PD)
    # print((PD >= MINIMUM_PD) and (PD <= MAXIMUM_PD))
    return (PD >= MINIMUM_PD) and (PD <= MAXIMUM_PD);


#Calculate thresholded saliency map to isolate black strip of card
#That method has also not been used in the final version of the code ..  .
def getSaliencyMap(pic):    
    (success, saliencyMap) = saliency.computeSaliency(pic)
    saliencyMap = np.uint8(saliencyMap * 255)  
    threshold = cv2.threshold(saliencyMap, 35, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(threshold, kernel, iterations=1)
    return erosion






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
