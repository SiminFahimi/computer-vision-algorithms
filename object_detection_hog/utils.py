import cv2
import numpy as np

def resize_image(image, scale):
    orig_h, orig_w = image.shape[:2]
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    return cv2.resize(image, (new_w, new_h))

def load_image(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not load image at {path}")
        return None
    return img

def knee_threshold(score_map):
    scores = np.sort(score_map.flatten())[::-1]
    n = len(scores)
    x = np.arange(n)
    y = scores
    p1 = np.array([0, y[0]])
    p2 = np.array([n - 1, y[-1]])
    den = max(np.linalg.norm(p2 - p1), 1e-10)
    distances = np.abs(np.cross(p2 - p1, np.vstack([x, y]).T - p1)) / den
    for i in range(1, len(distances) - 1):
        if distances[i] > distances[i-1] and distances[i] > distances[i+1]:
            knee_index = i
            break
    else:
        knee_index = np.argmax(distances)
    return scores[knee_index]

def nms(detections, iou_thresh=0.15):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if compute_iou(best, d) < iou_thresh]
    return keep

def compute_iou(box1, box2):
    ax, ay, aw, ah = box1["x"], box1["y"], box1["w"], box1["h"]
    bx, by, bw, bh = box2["x"], box2["y"], box2["w"], box2["h"]
    xi1 = max(ax, bx)
    yi1 = max(ay, by)
    xi2 = min(ax + aw, bx + bw)
    yi2 = min(ay + ah, by + bh)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union

def normalize(v):
    v = v.astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(img1, img2):
    return np.dot(normalize(img1), normalize(img2))

def compute_gradients(img):
    from common.kernel import Custom_kernel, add_filter
    sobel_x = Custom_kernel("sobel_x")
    sobel_y = Custom_kernel("sobel_y")
    gx = add_filter(img, sobel_x).astype(np.float32)
    gy = add_filter(img, sobel_y).astype(np.float32)
    magnitude = np.sqrt(gx**2 + gy**2 + 1e-6)
    orientation = np.arctan2(gy, gx) * 180 / np.pi
    orientation = np.mod(orientation, 180)
    return magnitude, orientation

from .hog_descriptor import *
from .utils import *
import cv2

def build_prototype(images, cell_size, block_size, num_bin): 
    prototypes = {}
    for label, img_list in images.items():
        all_feats_for_label = []
        for img_path in img_list:

            hog_features = hog_descriptor(cv2.imread(img_path, 0), cell_size, block_size, num_bin)
            all_feats_for_label.append(hog_features)

        prototypes[label] = np.mean(np.array(all_feats_for_label), axis=0)

    return prototypes

