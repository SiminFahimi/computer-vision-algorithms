:::writing
# Computer Vision Algorithms (From Scratch)

A collection of classic computer vision algorithms implemented **from scratch in Python**, focusing on understanding the mathematical foundations behind image processing and feature detection.

This project includes implementations of edge detection, corner detection, image pyramids, and object detection using gradient-based descriptors.

---

## Implemented Algorithms

### 1. Harris Corner Detection
Detects interest points (corners) in an image using the Harris response function.

Features:
- Gradient computation using Sobel filters
- Structure tensor calculation
- Corner response function
- Non‑maximum suppression for final keypoints

Output:
- Image with detected corner points.

---

### 2. Canny Edge Detection
A multi-stage edge detector designed to produce clean and thin edges.

Pipeline:
1. Gaussian smoothing
2. Gradient computation (Sobel filters)
3. Gradient magnitude and orientation
4. Non‑maximum suppression
5. Double threshold
6. Edge tracking by hysteresis

Output:
- Binary edge map.

---

### 3. Laplacian Pyramid
Multi‑scale image representation useful for image blending and compression.

Pipeline:
1. Gaussian pyramid generation
2. Laplacian pyramid construction
3. Reconstruction from pyramid levels

Output:
- Multi-scale representation of the input image.

---

### 4. HOG‑Based Object Detection
A simple object detection method based on **Histogram of Oriented Gradients (HOG)** descriptors and **cosine similarity**.

Pipeline:
1. Compute image gradients
2. Build HOG descriptors for cells
3. Block normalization
4. Sliding window search
5. Cosine similarity scoring with template descriptor
6. Non‑maximum suppression for final detections

Output:
- Bounding boxes around detected objects.

---

## Project Structure

```
computer-vision-algorithms/
│
├── run.py
│
├── common/
│   ├── __init__.py
│   └── kernel.py
│
├── harris_corner/
│   ├── harris.py
│   └── demo.py
│
├── canny_edge/
│   ├── canny.py
│   └── demo.py
│
├── laplacian_pyramid/
│   ├── pyramid.py
│   └── demo.py
│
├── object_detection_hog/
│   ├── gradients.py
│   ├── hog_descriptor.py
│   ├── similarity.py
│   ├── detector.py
│   └── demo.py
│
├── data/
│   └── sample_images
│
└── results/
```

---

## Installation

Clone the repository:

```
git clone https://github.com/SiminFahimi/computer-vision-algorithms.git
cd computer-vision-algorithms
```

Install required packages:

```
pip install numpy matplotlib opencv-python
```

---

## Running the Algorithms

A central script is provided to run each algorithm.

Example:

```
python run.py --method harris
```

Available options:

```
python run.py --method harris
python run.py --method canny
python run.py --method pyramid
python run.py --method detection
```

Each module also contains a **demo script** that demonstrates the algorithm independently.

Example:

```
python -m canny_edge.demo
```

---

## Example Results

Typical outputs include:

- Corner detection visualization
- Binary edge maps
- Multi‑scale pyramid levels
- Object detection bounding boxes

Example result images are stored in:

```
results/
```

---

## Goals of the Project

This repository focuses on:

- Understanding classical computer vision methods
- Implementing algorithms **without relying on high‑level libraries**
- Learning the mathematical intuition behind feature detection
- Building reusable computer vision modules

---

## Future Improvements

Possible extensions:

- HOG + SVM detector
- SIFT‑like keypoints
- RANSAC for feature matching
- Image stitching
- Optical flow
- Real‑time webcam demos
