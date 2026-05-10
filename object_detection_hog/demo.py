import cv2
from .utils import *
from .hog_object_detector import hog_multiscale_detector


A_imgs = ["data/a.png", "data/a1.png", "data/a2.png", "data/a3.png"]
B_imgs = ["data/b.png", "data/b1.png", "data/b2.png", "data/b3.png"]
C_imgs = ["data/c.png", "data/c1.png", "data/c2.png", "data/c3.png"]

image_path = "data/test.png"

def run_detector():

    cell_size= 4
    block_size= 2
    stride= 1
    num_bin = 15
    
    image = load_image(image_path, grayscale=True)
    if image is None:
        return

    prototypes = build_prototype({
        "A": A_imgs,
        "B": B_imgs,
        "C": C_imgs
    }, cell_size, block_size, num_bin)

    detections = hog_multiscale_detector(
        image,
        prototypes,
        cell_size= cell_size,
        block_size= block_size,
        stride= stride,
        num_bin= num_bin,
        scales=None
        )

    for label in ("A", "B", "C"):

        image_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        label_detections = [
            d for d in detections if d["label"] == label
        ]

        for d in label_detections:

            x, y, w, h = d["x"], d["y"], d["w"], d["h"]

            cv2.rectangle(
                image_vis,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )


        print(f"HOG detections for {label}: {len(label_detections)}")

        cv2.imwrite(
            f"results/HOG_detection_result_{label}.png",
            image_vis
        )

        cv2.imshow(f"HOG Detection {label}", image_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

run_detector()