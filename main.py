import argparse
from PIL import Image
import cv2 as cv
import numpy as np
from utils import guassian_kernel, non_max_sup, sobel_filter, hysteresis, threshold
from scipy.ndimage import convolve

def detect(img_path, threshold1, threshold2, save_file, sigma=1, kernel_size=5):
    # Open the image
    image = Image.open(img_path)
    
    # Convert image to grayscale
    gray_image = image.convert("L")
    gray_image_np = np.array(gray_image)
    
    # Generate Gaussian kernel
    guassian_ker = guassian_kernel(kernel_size, sigma)
    
    # Apply Gaussian blur
    guassian_blur = convolve(gray_image_np, guassian_ker)
    
    # Apply Sobel filter
    gradient, theta = sobel_filter(guassian_blur)
    
    # Apply Non-Maximum Suppression
    non_max_supression = non_max_sup(gradient, theta)
    
    # Apply Thresholding
    thresholding = threshold(non_max_supression, lowThresholdRatio=threshold1, hightThresholdRatio=threshold2)
    
    # Apply Hysteresis
    final_result = hysteresis(thresholding[0] , weak=thresholding[1] , strong=thresholding[2])
    
    # Save the final image
    cv.imwrite(save_file, final_result)
    print(f"Image saved in {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Detection using custom algorithms")
    
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--threshold1", type=float, required=True, help="Low threshold ratio for edge detection")
    parser.add_argument("--threshold2", type=float, required=True, help="High threshold ratio for edge detection")
    parser.add_argument("--save_file", type=str, required=True, help="Path to save the processed image")
    parser.add_argument("--sigma", type=float, default=1, help="Sigma for Gaussian kernel")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for Gaussian blur")
    
    args = parser.parse_args()
    
    detect(args.img_path, args.threshold1, args.threshold2, args.save_file, args.sigma, args.kernel_size)
    
    