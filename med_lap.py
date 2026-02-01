#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Corrected path — replace with your actual image file
img = cv2.imread(r"C:\Users\LapMac\OneDrive\Desktop\Kidney-Stone-Detection-IP-master\Images Used\image3.JPG", 0)  # 0 for grayscale

# Check if image loaded
if img is None:
    print("Error: Image not found or failed to load.")
    exit()

# Apply median blur
dst = cv2.medianBlur(img, 5)  # You can change 5 to any other odd number

# Calculate the Laplacian
lap = cv2.Laplacian(dst, cv2.CV_64F)

# Sharpening (you can tune the factor 0.3)
sharp = dst - 0.3 * lap
sharp = np.uint8(cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX))

# Histogram Equalization
equ = cv2.equalizeHist(sharp)

# Show original and processed images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('Output Image')
plt.axis('off')

plt.show()
