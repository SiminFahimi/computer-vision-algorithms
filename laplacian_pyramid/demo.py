"""laplacian_pyramid"""
def run_pyramid():
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2 as cv
    from . import laplacian_pyramid

    # Load image 
    image = cv.imread('data/lena.jpg', cv.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0

    # --- Build Laplacian Pyramid ---
    d, u, l = laplacian_pyramid.laplacian_pyramid(image)

    # Prepare canvases for visualization 
    total_width = sum(img.shape[1] for img in d)
    max_height = d[0].shape[0]

    canvas_gaussian = np.zeros((max_height, total_width))
    canvas_laplacian = np.zeros((max_height, total_width))

    # Gaussian Pyramid (Downsampling)
    x_offset = 0
    for img in d:
        h, w = img.shape
        canvas_gaussian[0:h, x_offset:x_offset+w] = np.abs(img)
        x_offset += w

    # Laplacian Pyramid (High-Frequency details)
    x_offset = 0
    for img in l:
        h, w = img.shape
        canvas_laplacian[0:h, x_offset:x_offset+w] = np.abs(img)
        x_offset += w

    # Display all three outputs in a single figure
    plt.figure(figsize=(12, 18))

    # 1. Gaussian Pyramid
    plt.subplot(3, 1, 1)
    plt.imshow(canvas_gaussian, cmap='gray')
    plt.title("Gaussian Pyramid (Downsampling)")
    plt.axis('off')

    # 2. Laplacian Pyramid
    plt.subplot(3, 1, 2)
    plt.imshow(np.clip(canvas_laplacian, 0, 1), cmap='gray')
    plt.title("Laplacian Pyramid (High-Frequency Details)")
    plt.axis('off')

    # 3. Reconstructed Image (Up-sampling)
    plt.subplot(3, 1, 3)
    plt.imshow(u[0], cmap='gray')
    plt.title("Reconstructed Image (Up-sampling)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/laplacian_pyramid.png")
    plt.show()

run_pyramid()