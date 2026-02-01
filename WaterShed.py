#%%     # this line is for VSCode editor only
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("Images Used/image2.JPG")
if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(12, 6))
plt.subplot(231), plt.imshow(thresh, cmap='gray')
plt.title('Thresh: Binary + OTSU'), plt.xticks([]), plt.yticks([])

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

plt.subplot(232), plt.imshow(sure_fg, cmap='gray')
plt.title('Sure Foreground'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(unknown, cmap='gray')
plt.title('Unknown Region'), plt.xticks([]), plt.yticks([])

# Marker labeling
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

plt.subplot(234), plt.imshow(markers, cmap='jet')
plt.title('Markers'), plt.xticks([]), plt.yticks([])

# Apply Watershed
# Use the original image for watershed
img = cv2.imread("Images Used/image2.JPG")
img = cv2.medianBlur(img, 5)
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]  # Mark boundaries in red

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(235), plt.imshow(img_rgb)
plt.title('Watershed Result'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
