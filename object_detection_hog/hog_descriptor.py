import numpy as np
from .gradients import compute_gradients


def hog_descriptor(img, cell_size=8, bins=9):

    magnitude, orientation = compute_gradients(img)

    h, w = img.shape
    cells_x = w // cell_size
    cells_y = h // cell_size

    hist = np.zeros((cells_y, cells_x, bins))

    bin_width = 180 / bins

    for y in range(cells_y):
        for x in range(cells_x):

            cell_mag = magnitude[
                y*cell_size:(y+1)*cell_size,
                x*cell_size:(x+1)*cell_size
            ]

            cell_ori = orientation[
                y*cell_size:(y+1)*cell_size,
                x*cell_size:(x+1)*cell_size
            ]

            for i in range(cell_size):
                for j in range(cell_size):

                    bin_idx = int(cell_ori[i,j] // bin_width)
                    hist[y,x,bin_idx] += cell_mag[i,j]

    return hist.flatten()
