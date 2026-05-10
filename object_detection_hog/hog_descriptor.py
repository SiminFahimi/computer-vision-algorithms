import numpy as np
from .utils import compute_gradients

def hog_descriptor(image, cell_size, block_size, num_bins):
    magnitude, orientation = compute_gradients(image)
    h, w = image.shape

    # Ensure dimensions are multiples of cell_size
    h_crop = (h // cell_size) * cell_size
    w_crop = (w // cell_size) * cell_size
    magnitude = magnitude[:h_crop, :w_crop]
    orientation = orientation[:h_crop, :w_crop]
    
    cells_y = h_crop // cell_size
    cells_x = w_crop // cell_size
    
    # Reshape to cells
    mag_cells = magnitude.reshape(cells_y, cell_size, cells_x, cell_size)
    orient_cells = orientation.reshape(cells_y, cell_size, cells_x, cell_size)
    
    # Calculate bin indices
    bin_width = 180.0 / num_bins
    orient_flat = orient_cells.reshape(-1, cell_size, cell_size)
    
    # Initialize histogram
    hist = np.zeros((num_bins, cells_y, cells_x), dtype=np.float32)
    
    # Compute histograms using vectorized operations
    for i in range(cells_y):
        for j in range(cells_x):
            cell_mag = mag_cells[i, :, j, :].flatten()
            cell_orient = orient_cells[i, :, j, :].flatten()
            
            bin_idx = np.floor(cell_orient / bin_width).astype(int) % num_bins
            bin_weights = np.abs(cell_orient - (bin_idx * bin_width)) / bin_width
            
            # Add to histogram with interpolation
            for b in range(num_bins):
                mask = (bin_idx == b)
                hist[b, i, j] += np.sum(cell_mag[mask] * (1 - bin_weights[mask]))
                
                mask_next = (bin_idx == (b - 1) % num_bins)
                hist[b, i, j] += np.sum(cell_mag[mask_next] * bin_weights[mask_next])
    
    # Block normalization
    features = []
    for by in range(cells_y - block_size + 1):
        for bx in range(cells_x - block_size + 1):
            block = hist[:, by:by+block_size, bx:bx+block_size].flatten()
            block_norm = block / (np.linalg.norm(block) + 1e-6)
            block_norm = np.clip(block_norm, 0, 0.2)
            block_norm = block_norm / (np.linalg.norm(block_norm) + 1e-6)
            features.append(block_norm)
    features = np.array(features).reshape(
    (cells_y - block_size + 1, cells_x - block_size + 1, block_size * block_size * num_bins)
    )
    
    return features