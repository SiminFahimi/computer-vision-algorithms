from .utils import *
from .hog_descriptor import *
from .classifier import cosine_classifier

def hog_object_detector(image, prototypes, cell_size, block_size, stride, num_bin):

    image_hog = hog_descriptor(image, cell_size, block_size, num_bin)

    i_h, i_w, _ = image_hog.shape

    detections = []

    for label, proto in prototypes.items():

        p_h, p_w, _ = proto.shape

        for y in range(0, i_h - p_h + 1, stride):
            for x in range(0, i_w - p_w + 1, stride):

                window = image_hog[y:y+p_h, x:x+p_w]

                if window.shape != proto.shape:
                    continue

                best_label, best_score = cosine_classifier(
                    window,
                    {label: proto}
                )
                if best_score > 0.92:

                    x_real = x * cell_size * stride
                    y_real = y * cell_size * stride

                    detections.append({
                        "x": x_real,
                        "y": y_real,
                        "w": p_w * cell_size * block_size,
                        "h": p_h * cell_size * block_size,
                        "label": best_label,
                        "score": best_score
                    })

    return detections

def hog_multiscale_detector(image, prototypes, cell_size, block_size, stride, num_bin, scales=None):
    if scales is None:
        scales = [0.7, 0.8, 1.0, 1.2]

    detections = []
    
    for scale in scales:
        
        resized_img = resize_image(image, scale)
        coords = hog_object_detector(resized_img, prototypes, cell_size, block_size, stride,num_bin)

        for d in coords:
            detections.append({
                "x": int(d["x"] / scale),
                "y": int(d["y"] / scale),
                "w": int(d["w"] / scale),
                "h": int(d["h"] / scale),
                "score": d["score"],
                "label": d["label"]
            })

    return nms(detections)