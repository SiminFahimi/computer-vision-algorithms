import cv2
import numpy as np
from .detector import *

def combine_templates(template_list):
    if not template_list:
        return None
    base_h, base_w = template_list[0].shape[:2]
    resized = [cv2.resize(t, (base_w, base_h)) for t in template_list]
    return np.mean(np.stack(resized), axis=0).astype(np.float32)


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

        coords = detect(resized_img, template, weight_mask, threshold=0.82, stride=2)

        # Scale back to original coordinates
        for (y, x) in coords:
            detections.append((int(y / scale), int(x / scale)))

    return detections


def run_detector():
    image = cv2.imread("data/test.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load test.png")
        return

    template_paths = ["data/a.png", "data/a1.png"]
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

    for (y, x) in detections:
        cv2.rectangle(image_vis, (x, y), (x + t_w, y + t_h), (0, 255, 0), 2)

    cv2.imshow("Character Detection - Weighted NCC", image_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Total detections: {len(detections)}")
