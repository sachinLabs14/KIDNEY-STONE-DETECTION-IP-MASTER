# Kidney Stone Detection using Image Processing

This project provides several Python scripts for detecting kidney stones in medical images using various image processing techniques.

## Implemented Features

- **Gabor Filter**: For texture-based feature extraction.
- **Histogram Equalization**: To improve image contrast.
- **Laplacian & Sobel Filters**: For edge detection and sharpening.
- **Watershed Segmentation**: For precise object boundary detection.
- **Noise Reduction**: Implementation of Gaussian and Median blurring.
- **Grayscale Conversion**: Preprocessing step for most filters.

## Requirements

To run the scripts, you need to install the following dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `opencv-python`
- `numpy`
- `matplotlib`

## How to Run

1. Ensure you have the images in the `Images Used/` directory.
2. Run any of the feature scripts (e.g., `python Finalcode.py`).
