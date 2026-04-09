"""character detection"""

import cv2
from .detector import detect_object


def run_detector():

    image = cv2.imread("data/test.jpg")
    template = cv2.imread("data/a.jpg")

    detections = detect_object(image, template)

    for (x,y,w,h,s) in detections:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
