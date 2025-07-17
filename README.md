# Road Extraction from Satellite Images

A Python tool for extracting road networks from satellite imagery using Canny edge detection and morphological operations.

## Features

- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for improved visibility
- **Edge Detection**: Canny edge detection with optional bilateral filtering
- **Morphological Processing**: Advanced operations to isolate road-like features
- **Noise Reduction**: Automatic removal of small components and building-like structures
- **Visual Output**: Side-by-side comparison of all processing steps

## Requirements

```bash
pip install opencv-python numpy matplotlib
```

## Usage

```bash
python road.py image.tif
```

The program will automatically process your satellite image and save the results in the same directory.

## Output Files

1. **`{filename}_roads.png`** - Composite image showing all processing steps
2. **`{filename}_road_mask.png`** - Binary mask of extracted roads

## Parameters

All parameters are configurable at the beginning of the script:

### Contrast Enhancement
- `CLAHE_CLIP_LIMIT` (default: 3.0) - Contrast limiting threshold
- `CLAHE_TILE_SIZE` (default: 8x8) - Grid size for adaptive histogram equalization

### Edge Detection
- `CANNY_LOW_THRESHOLD` (default: 10) - Lower threshold for edge detection
- `CANNY_HIGH_THRESHOLD` (default: 30) - Upper threshold for edge detection
- `USE_BILATERAL_FILTER` (default: True) - Enable edge-preserving smoothing

### Morphological Operations
- `MORPH_KERNEL_SIZE` (default: 3) - Kernel size for morphological operations
- `DILATE_ITERATIONS` (default: 2) - Iterations for connecting road segments
- `MIN_COMPONENT_SIZE` (default: 100) - Minimum pixel area to keep

## Algorithm Pipeline

1. **Preprocessing**: Convert to grayscale and enhance contrast using CLAHE
2. **Edge Detection**: Apply bilateral filter + Gaussian blur + Canny edge detection
3. **Morphological Processing**:
   - Dilation to connect nearby edges
   - Closing to fill gaps
   - Opening to remove noise
   - Top-hat transform to extract linear features
4. **Post-processing**: Remove small connected components

## Example Results

The output shows three stages of processing:

| Step 1: Contrast Enhanced | Step 2: Canny Edges | Step 3: Extracted Roads |
|--------------------------|--------------------|-----------------------|
| CLAHE applied to original | Edge detection result | Final road network |

## Technical Details

- **CLAHE**: Enhances local contrast while preventing over-amplification
- **Bilateral Filter**: Preserves edges while smoothing noise
- **Top-hat Transform**: Extracts bright linear features (roads) from background
- **Connected Components**: Filters results based on size to remove noise


