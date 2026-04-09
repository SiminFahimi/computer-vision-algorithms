import cv2
import numpy as np
from canny_edge import canny

from .hog_descriptor import hog_descriptor
from .similarity import cosine_similarity, edge_weighted_score

def edge_density(img):

    edges = canny.canny_edge_detection(img)
    return np.sum(edges) / edges.size


def detect_object(image, template, window_size=(64,128), stride=16, threshold=0.5):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_desc = hog_descriptor(template_gray)

    h, w = gray.shape
    win_w, win_h = window_size

    detections = []

    for y in range(0, h-win_h, stride):
        for x in range(0, w-win_w, stride):

            window = gray[y:y+win_h, x:x+win_w]

            desc = hog_descriptor(window)

            cos = cosine_similarity(template_desc, desc)

            edge_score = edge_density(window)

            score = edge_weighted_score(cos, edge_score)

            if score > threshold:
                detections.append((x,y,win_w,win_h,score))

    return detections
