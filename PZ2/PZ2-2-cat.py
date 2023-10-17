import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

# Read local image
img = io.imread("snow_cat.jpg")

# Convert color scheme to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Init HSV min and max threshold values
MIN_H, MIN_S, MIN_V = 0, 0, 0
MAX_H, MAX_S, MAX_V = 180, 255, 255

# Calculate histograms for each channel respectively
# H: 0, S: 1, V: 2
hist_h = cv2.calcHist([hsv_img], [0], None, [MAX_H], [MIN_H, MAX_H])
hist_s = cv2.calcHist([hsv_img], [1], None, [MAX_S], [MIN_S, MAX_S])
hist_v = cv2.calcHist([hsv_img], [2], None, [MAX_V], [MIN_V, MAX_V])

# Print histograms if needed
if False:
    plt.figure('Hsv histogram')
    plt.plot(hist_h, color='red')
    plt.plot(hist_s, color='green')
    plt.plot(hist_v, color='blue')
    plt.show()

# Hard-coded threshold values of Saturation channel
min_hsv_s, max_hsv_s = 15, 30

# Threshold image removing pixels out of range of the Saturation threshold
threshold_img = cv2.inRange(hsv_img, (MIN_H, min_hsv_s, MIN_V), (MAX_H, max_hsv_s, MAX_V))  # gives grayscale img with 255 and 0 values

# Canny filter method
blurred_img = cv2.GaussianBlur(threshold_img, (5, 5), 0)
canny_img = cv2.Canny(blurred_img, 100, 200)

# Subtraction method
blurred_alot_img = cv2.GaussianBlur(threshold_img, (49, 49), 0)
subtracted_img = cv2.subtract(threshold_img, blurred_alot_img)

# Plot results
plt.figure('Results')

# Plot first method
plt.subplot(2, 4, 1), plt.imshow(img, aspect='auto'), plt.title('Original image')
plt.subplot(2, 4, 2), plt.imshow(threshold_img, cmap='gray', aspect='auto'), plt.title('Grayscale threshold image')
plt.subplot(2, 4, 3), plt.imshow(blurred_img, cmap='gray', aspect='auto'), plt.title('Blurred image')
plt.subplot(2, 4, 4), plt.imshow(canny_img, cmap='gray', aspect='auto'), plt.title('Edges via Canny image')

# Plot second method
plt.subplot(2, 4, 5), plt.imshow(img, aspect='auto'), plt.title('Original image')
plt.subplot(2, 4, 6), plt.imshow(threshold_img, cmap='gray', aspect='auto'), plt.title('Grayscale threshold image')
plt.subplot(2, 4, 7), plt.imshow(blurred_alot_img, cmap='gray', aspect='auto'), plt.title('Blurred a lot image')
plt.subplot(2, 4, 8), plt.imshow(subtracted_img, cmap='gray', aspect='auto'), plt.title('Subtracted image')

plt.show()

# Terminate plots
plt.close()