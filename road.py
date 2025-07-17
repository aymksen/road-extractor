#!/usr/bin/env python3
"""
Road extraction from satellite images using Canny-morphology pipeline
Usage: python road.py image.tif

Requirements:
    pip install opencv-python numpy matplotlib

Author: Road Detection Script
"""

import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

# ==============================================================================
# EDITABLE PARAMETERS - Adjust these values to optimize road detection
# ==============================================================================

# Contrast enhancement parameters
CLAHE_CLIP_LIMIT = 3.0          # Contrast limiting threshold (1.0-5.0)
CLAHE_TILE_SIZE = (8, 8)        # Grid size for adaptive histogram equalization

# Gaussian blur parameters
GAUSSIAN_KERNEL_SIZE = (3, 3)   # Kernel size for noise reduction
GAUSSIAN_SIGMA = 0.5             # Standard deviation for Gaussian blur

# Canny edge detection parameters
CANNY_LOW_THRESHOLD = 10         # Lower threshold for edge detection
CANNY_HIGH_THRESHOLD = 30        # Upper threshold for edge detection

# Morphological operation parameters
MORPH_KERNEL_SIZE = 3            # Kernel size for morphological operations
MORPH_ITERATIONS = 1             # Number of iterations for morphological ops
DILATE_ITERATIONS = 2            # Dilation to connect road segments
ERODE_ITERATIONS = 1             # Erosion to thin roads

# Line detection parameters
MIN_LINE_LENGTH = 50             # Minimum length of detected lines (pixels)
MAX_LINE_GAP = 20                # Maximum gap between line segments (pixels)

# Additional filtering parameters
USE_BILATERAL_FILTER = True     # Use bilateral filter for edge-preserving smoothing
BILATERAL_D = 9                  # Diameter of bilateral filter
BILATERAL_SIGMA_COLOR = 75       # Color sigma for bilateral filter
BILATERAL_SIGMA_SPACE = 75       # Space sigma for bilateral filter

# Alternative edge detection
USE_ADAPTIVE_THRESHOLD = False   # Use adaptive thresholding instead of Canny
ADAPTIVE_BLOCK_SIZE = 11         # Block size for adaptive threshold
ADAPTIVE_C = 2                   # Constant for adaptive threshold

# Output parameters
OUTPUT_DPI = 150                 # DPI for output image
FIGURE_SIZE = (15, 5)            # Figure size in inches (width, height)

# ==============================================================================

def enhance_contrast(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if len(image.shape) == 3:
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    enhanced = clahe.apply(gray)
    
    return enhanced

def detect_edges(image):
    """Apply Gaussian blur and Canny edge detection"""
    # Optional: Apply bilateral filter for edge-preserving smoothing
    if USE_BILATERAL_FILTER:
        filtered = cv2.bilateralFilter(image, BILATERAL_D, 
                                     BILATERAL_SIGMA_COLOR, 
                                     BILATERAL_SIGMA_SPACE)
    else:
        filtered = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(filtered, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    
    if USE_ADAPTIVE_THRESHOLD:
        # Alternative: Use adaptive thresholding
        edges = cv2.adaptiveThreshold(blurred, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV,
                                    ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    else:
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
    return edges

def extract_roads(edges):
    """Apply morphological operations to extract road-like features"""
    # Create morphological kernels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                      (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    # Create a larger kernel for removing building-like structures
    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    # First, dilate to connect nearby edges (road segments)
    dilated = cv2.dilate(edges, kernel, iterations=DILATE_ITERATIONS)
    
    # Close gaps in edges
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, 
                             iterations=MORPH_ITERATIONS)
    
    # Remove large blob-like structures (likely buildings)
    # Use opening with a larger kernel
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, large_kernel, 
                              iterations=1)
    
    # Apply a top-hat transform to extract linear features
    tophat = cv2.morphologyEx(closed, cv2.MORPH_TOPHAT, large_kernel)
    
    # Combine the cleaned image with tophat to enhance roads
    roads = cv2.add(cleaned, tophat)
    
    # Thin the roads
    thinned = cv2.erode(roads, kernel, iterations=ERODE_ITERATIONS)
    
    # Apply threshold to clean up
    _, binary = cv2.threshold(thinned, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def detect_lines(road_image):
    """Detect lines in the processed image using HoughLinesP"""
    # Create output image
    lines_image = np.zeros_like(road_image)
    
    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(road_image, 
                           rho=1, 
                           theta=np.pi/180, 
                           threshold=50,
                           minLineLength=MIN_LINE_LENGTH,
                           maxLineGap=MAX_LINE_GAP)
    
    # Draw lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
    
    return lines_image

def process_image(image_path):
    """Main processing pipeline"""
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Step 1: Enhance contrast
    print("Step 1: Enhancing contrast...")
    enhanced = enhance_contrast(image)
    
    # Step 2: Edge detection
    print("Step 2: Detecting edges...")
    edges = detect_edges(enhanced)
    
    # Step 3: Extract roads
    print("Step 3: Extracting road features...")
    roads = extract_roads(edges)
    
    # Step 4: Detect lines (optional refinement)
    print("Step 4: Refining road lines...")
    road_lines = detect_lines(roads)
    
    return enhanced, edges, road_lines

def save_results(original_path, enhanced, edges, roads):
    """Create and save the output image with all processing steps"""
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    
    # Plot contrast enhanced image
    axes[0].imshow(enhanced, cmap='gray')
    axes[0].set_title('Step 1: Contrast Enhanced')
    axes[0].axis('off')
    
    # Plot Canny edges
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Step 2: Canny Edges')
    axes[1].axis('off')
    
    # Plot extracted roads
    axes[2].imshow(roads, cmap='gray')
    axes[2].set_title('Step 3: Extracted Roads')
    axes[2].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    output_dir = os.path.dirname(original_path)
    output_path = os.path.join(output_dir, f"{base_name}_roads.png")
    
    # Save figure
    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Output saved to: {output_path}")
    
    # Also save individual road mask
    road_mask_path = os.path.join(output_dir, f"{base_name}_road_mask.png")
    cv2.imwrite(road_mask_path, roads)
    print(f"Road mask saved to: {road_mask_path}")

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python road.py image.tif")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found")
        sys.exit(1)
    
    print(f"Processing: {image_path}")
    print("-" * 50)
    
    # Process image
    result = process_image(image_path)
    
    if result is not None:
        enhanced, edges, roads = result
        
        # Save results
        save_results(image_path, enhanced, edges, roads)
        
        print("-" * 50)
        print("Processing complete!")
    else:
        print("Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()