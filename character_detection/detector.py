import numpy as np
from canny_edge import canny 
from common import kernel as krn

def weighted_NCC(template, image, weight_mask, weighted_mean_t, weighted_mean_i):
    t_zero_mean = template - weighted_mean_t
    i_zero_mean = image - weighted_mean_i

    ncc_num = np.sum(weight_mask * t_zero_mean * i_zero_mean)
    t_var = np.sum(weight_mask * (t_zero_mean ** 2))
    i_var = np.sum(weight_mask * (i_zero_mean ** 2))

    denominator = np.sqrt(max(t_var, 1e-10) * max(i_var, 1e-10))
    return ncc_num / denominator


def build_weight_mask(template):

    edges = canny.canny_edge_detection(template)[0]

    edges = edges.astype(np.float32)

    g = krn.Gaussian_kernel(1.5, (7,7))

    weight = krn.add_filter(edges, g)

    weight += 1e-6

    weight = weight / np.sum(weight)
    return weight


def detect(img, template, weight_mask, threshold=0.4, stride=1):
    img = np.array(img, dtype=np.float32)
    template = np.array(template, dtype=np.float32)
    weight_mask = np.array(weight_mask, dtype=np.float32)

    t_h, t_w = template.shape
    i_h, i_w = img.shape

    weighted_mean_t = np.sum(weight_mask * template)
    coordinates = []
    max_score = -1

    for i in range(0, i_h - t_h + 1, stride):
        for j in range(0, i_w - t_w + 1, stride):
            cropped = img[i:i+t_h, j:j+t_w]
            weighted_mean_i = np.sum(weight_mask * cropped)

            score = weighted_NCC(template, cropped, weight_mask, weighted_mean_t, weighted_mean_i)

            if score > max_score:
                max_score = score

            if score > threshold:
                coordinates.append((i, j))
                
    print("MAX SCORE:", max_score)

    return coordinates