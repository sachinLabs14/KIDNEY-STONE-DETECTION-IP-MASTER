#%%     # this line is for VSCode editor only
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Change this path to your actual image location
image_path = r'C:\Users\LapMac\OneDrive\Desktop\Kidney-Stone-Detection-IP-master\images\image1.jpg'

# Load image in grayscale
img = cv2.imread(image_path, 0)
if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

def build_filters():
    """Returns a list of Gabor kernels in several orientations."""
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {
            'ksize': (ksize, ksize), 'sigma': 2.5, 'theta': theta,
            'lambd': 15.0, 'gamma': 0.02, 'psi': 0, 'ktype': cv2.CV_32F
        }
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5 * kern.sum()
        filters.append((kern, params))
    return filters

def process(img, filters):
    """Applies each Gabor filter to the image and keeps the max response."""
    accum = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

# Main pipeline
filters = build_filters()
gabor_output = process(img, filters)

# Histogram equalization
equ = cv2.equalizeHist(gabor_output)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(gabor_output, cmap='gray')
plt.title('Gabor Only'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(equ, cmap='gray')
plt.title('Histogram + Gabor'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
