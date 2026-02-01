#%%  # for VSCode cell execution

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (color image, flag=1)
image = cv2.imread(r'C:\Users\LapMac\OneDrive\Desktop\Kidney-Stone-Detection-IP-master\Images Used\image3.JPG', 1)

# Check if image loaded successfully
if image is None:
    print("Error: Image not found or cannot be opened.")
    exit()

# Apply Sobel filter in Y direction
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# Display Sobel Y result
plt.subplot(111)
plt.imshow(np.uint8(np.absolute(sobely)), cmap='gray')
plt.title('Sobel Y')
plt.axis('off')
plt.show()  
