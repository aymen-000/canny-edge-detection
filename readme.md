# Canny Edge Detection

This project implements a custom Canny Edge Detection algorithm using Python and various image processing techniques. The implementation includes Gaussian Blur, Sobel filtering, Non-Maximum Suppression, Thresholding, and Hysteresis. The project uses OpenCV and SciPy for image manipulation and edge detection.

## Features

- Convert input images to grayscale.
- Apply Gaussian Blur to smooth the image and reduce noise.
- Detect edges using Sobel filters.
- Suppress non-maximum edges.
- Apply double thresholding and edge tracking by hysteresis.
- Save the final processed image to a specified file.
  
## Results 
![original](/imgs/2.jpg)
![result](/imgs/result1.jpg)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/canny-edge-detection.git
   cd canny-edge-detection
   pip install -r requirements.txt
   python main.py --img_path ./imgs/your_image.jpg --threshold1 0.05 --threshold2 0.09 --save_file ./imgs/result.jpg

## Arguments 

- img_path: Path to the input image.
- threshold1: Low threshold ratio for edge detection.
- threshold2: High threshold ratio for edge detection.
- save_file: Path to save the processed image.
- sigma: Sigma value for Gaussian kernel (default: 1).
- kernel_size: Kernel size for Gaussian blur (default: 5).