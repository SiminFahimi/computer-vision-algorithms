import cv2
import numpy as np
from .detector import *

def combine_templates(template_list):
    if not template_list:
        return None
    base_h, base_w = template_list[0].shape[:2]
    resized = [cv2.resize(t, (base_w, base_h)) for t in template_list]
    return np.mean(np.stack(resized), axis=0).astype(np.float32)

def nms(detections, iou_thresh=0.15):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        detections = [
            d for d in detections
            if compute_iou(best, d) < iou_thresh
        ]
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
    
def detect_multiscale(image, template, scales=None):
    if scales is None:
        scales = [0.7, 0.85, 1.0, 1.2, 1.3]

    detections = []
    orig_h, orig_w = image.shape[:2]
    t_h, t_w = template.shape[:2]

    for scale in scales:
        
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        resized_img = cv2.resize(image, (new_w, new_h))

        print(f"Searching at scale {scale:.2f}...")

        weight_mask = build_weight_mask(template)   

        coords = detect(resized_img, template, weight_mask, stride=1)

        # Scale back to original coordinates
        for (y, x, score) in coords:
            detections.append({
                "x": int(x / scale),
                "y": int(y / scale),
                "w": t_w,
                "h": t_h,
                "score": score
            })


    detections = nms(detections)
    return detections


def run_detector():
    image = cv2.imread("data/test.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load test.png")
        return

    #template_paths = ["data/a.png", "data/a1.png", "data/a2.png", "data/a3.png"]
    #template_paths = ["data/b.png", "data/b1.png", "data/b2.png", "data/b3.png"]
    template_paths = ["data/c.png", "data/c1.png", "data/c2.png", "data/c3.png"]


    templates = [cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) 
                 for p in template_paths if cv2.imread(p) is not None]

    combined_template = combine_templates(templates)
    if combined_template is None:
        print("Error: Could not combine templates")
        return

    detections = detect_multiscale(image, combined_template)

    # Visualization
    image_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    t_h, t_w = combined_template.shape[:2]

    for d in detections:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cv2.rectangle(image_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("Character Detection - Weighted NCC", image_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("results/character_detection_result.png",image_vis)
    print(f"Total detections: {len(detections)}")

run_detector()