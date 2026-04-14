# Computer Vision Algorithms (From Scratch)

Note: This project was built purely as a personal exercise to better understand and gain intuition about classical computer vision algorithms. It is not intended as a production-level system.

A collection of classic computer vision algorithms implemented **from scratch in Python** using only NumPy and basic OpenCV for image I/O and visualization.

The goal of this project is to deeply understand the mathematical foundations behind fundamental computer vision techniques by implementing them without relying on high-level OpenCV functions.

---

## Implemented Algorithms

### 1. Harris Corner Detection
Detects interest points (corners) using the Harris response function.

**Key Features:**
- Sobel gradient computation
- Gaussian smoothing of structure tensor components
- Harris response calculation
- Non-maximum suppression

**Module:** `harris_corner/`

---

### 2. Canny Edge Detection
A complete multi-stage edge detection pipeline.

**Pipeline:**
1. Gaussian smoothing  
2. Gradient computation (Sobel filters)  
3. Non-maximum suppression  
4. Double thresholding  
5. Edge tracking by hysteresis  

**Module:** `canny_edge/`

---

### 3. Laplacian Pyramid
Multi-scale image representation using Gaussian and Laplacian pyramids.

**Features:**
- Gaussian pyramid construction
- Laplacian pyramid computation
- Image reconstruction
- Visualization of pyramid levels

**Module:** `laplacian_pyramid/`

---

### 4. Character / Object Detection (Template Matching)
Sliding-window based detection using template matching with optional edge-weighted scoring.

**Features:**
- Multi-scale detection support
- Weighted Normalized Cross-Correlation (NCC)
- Edge-aware weighting using Canny detector
- Template averaging from multiple samples
- Non-Maximum Suppression (NMS)

**Module:** `character_detection/`

---

## Project Structure

```
computer-vision-algorithms/
├── run.py                          # Main runner script
├── common/
│   └── kernel.py                   # Gaussian & Sobel kernels + convolution
├── harris_corner/
│   ├── harris.py
│   └── demo.py
├── canny_edge/
│   ├── canny.py
│   └── demo.py
├── laplacian_pyramid/
│   ├── laplacian_pyramid.py
│   └── demo.py
├── character_detection/
│   ├── detector.py
│   └── demo.py
├── data/                           # Sample images (lena, square, test.png, characters, etc.)
└── README.md
```

## Requirements

```bash
pip install numpy matplotlib opencv-python
```


## How to Run
Using the central runner:
Bash

# Harris Corner Detection
```bash
python -m harris_corner.demo
```

# Canny Edge Detection
```bash
python -m canny_edge.demo
```

# Laplacian Pyramid
```bash
python -m laplacian_pyramid.demo
```

# Character Detection
```bash
python -m character_detection.demo
```


Goals of the Project

Implement classical computer vision algorithms from scratch (no high-level OpenCV functions like cv2.Canny or cv2.cornerHarris)
Understand the mathematical foundations behind each method
Build modular and reusable code structure
Visualize intermediate and final results clearly

Future Improvements (Ideas)

Add full HOG + SVM-based object detection
Improve Non-Maximum Suppression (NMS)
Implement SIFT-like keypoint detection
Optical flow implementation
Image stitching / panorama creation
Real-time webcam demos

## Example Results

Here are some sample outputs from the implemented algorithms:

### Harris Corner Detection vs Original

| Original Image | Detected Corners |
|----------------|------------------|
| ![Original](data/square.jpg) | ![Harris](results/harris_corner_detection.png) |

### Canny Edge Detection
| Original Image | Detected Edges |


|----------------|------------------|
| ![Original](data/lena.jpg) | ![Canny Edge Detection](results/canny_edge_detection.png) |


### Laplacian Pyramid
![Laplacian Pyramid - Gaussian & Laplacian Levels](results/laplacian_pyramid.png)

### Character Detection
![Detection of  character "a"](results/detection_result0.png)
![Detection of character "b"](results/detection_result1.png)
![Detection of character "c"](results/detection_result2.png)
