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


import numpy as np

def knee_threshold(score_map):
    scores = np.sort(score_map.flatten())[::-1]
    n = len(scores)

    x = np.arange(n)
    y = scores

    p1 = np.array([0, y[0]])
    p2 = np.array([n - 1, y[-1]])
    den = max(np.linalg.norm(p2 - p1), 1e-10)

    distances = np.abs(np.cross(p2 - p1, np.vstack([x, y]).T - p1)) / den

    d1 = np.diff(distances)

    for i in range(1, len(distances) - 1):
        if distances[i] > distances[i-1] and distances[i] > distances[i+1]:
            knee_index = i
            break
    else:
        knee_index = np.argmax(distances)

    threshold = scores[knee_index]
    return threshold


def detect(img, template, weight_mask, stride=1):
    img = np.array(img, dtype=np.float32)
    template = np.array(template, dtype=np.float32)
    weight_mask = np.array(weight_mask, dtype=np.float32)

    t_h, t_w = template.shape
    i_h, i_w = img.shape

    weighted_mean_t = np.sum(weight_mask * template)

    ys = range(0, i_h - t_h + 1, stride)
    xs = range(0, i_w - t_w + 1, stride)

    scores = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for y, i in enumerate(ys):
        for x, j in enumerate(xs):

            cropped = img[i:i+t_h, j:j+t_w]
            weighted_mean_i = np.sum(weight_mask * cropped)
            score = weighted_NCC(template, cropped, weight_mask, weighted_mean_t, weighted_mean_i)
            scores[y, x] = score

    threshold=knee_threshold(scores)
    
    ys, xs = np.where(scores >= max(threshold, 0.8))

    coordinates = [(int(y*stride), int(x*stride), scores[y, x]) for y, x in zip(ys, xs)]

    return coordinates