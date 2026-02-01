import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Set parameters
ddepth = cv.CV_16S
kernel_size = 3

# ✅ Load the actual image (Update the path if needed)
src = cv.imread("Images Used/image2.JPG")

# ✅ Check if image is loaded
if src is None:
    print("Error: Could not load image.")
    sys.exit()

# ✅ Remove noise using Gaussian blur
src = cv.GaussianBlur(src, (3, 3), 0)

# ✅ Convert to grayscale
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# ✅ Apply Laplacian
dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)

# ✅ Convert back to uint8
abs_dst = cv.convertScaleAbs(dst)

# ✅ Display using matplotlib
plt.subplot(121), plt.imshow(src_gray, cmap='gray')
plt.title('Grayscale Input'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(abs_dst, cmap='gray')
plt.title('Laplacian Output'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
