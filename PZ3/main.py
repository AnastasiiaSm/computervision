import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# Init constants
MIN_H, MIN_S, MIN_V = 0, 0, 0
MAX_H, MAX_S, MAX_V = 180, 255, 255
CALIBRATION_WINDOW_NAME = 'HSV mask'
def nothing(x):
    pass

should_calculate_histograms = False
should_show_calibration = False

# Read local image
img = cv2.imread('dz_pz/img_7.jpeg')
#img_arr = np.load('dz_pz/img_1.npz')['frame'] # I cannot make this work ;c (c) fanglores
#img = Image.fromarray(img_arr)

# Convert color scheme to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lowH = 0
highH = 180
lowS = 0
highS = 10
lowV = 200
highV = 255

lower_hsv_mask = np.array([lowH, lowS, lowV])
higher_hsv_mask = np.array([highH, highS, highV])

if should_show_calibration:
    cv2.namedWindow(CALIBRATION_WINDOW_NAME, 1)

    # create trackbars for high and low H, S, V channels
    cv2.createTrackbar('low H', CALIBRATION_WINDOW_NAME, lowH, MAX_H, nothing)
    cv2.createTrackbar('high H', CALIBRATION_WINDOW_NAME, highH, MAX_H, nothing)

    cv2.createTrackbar('low S', CALIBRATION_WINDOW_NAME, lowS, MAX_S, nothing)
    cv2.createTrackbar('high S', CALIBRATION_WINDOW_NAME, highS, MAX_S, nothing)

    cv2.createTrackbar('low V', CALIBRATION_WINDOW_NAME, lowV, MAX_V, nothing)
    cv2.createTrackbar('high V', CALIBRATION_WINDOW_NAME, highV, MAX_V, nothing)

    while True:
        lowH = cv2.getTrackbarPos('low H', CALIBRATION_WINDOW_NAME)
        highH = cv2.getTrackbarPos('high H', CALIBRATION_WINDOW_NAME)
        lowS = cv2.getTrackbarPos('low S', CALIBRATION_WINDOW_NAME)
        highS = cv2.getTrackbarPos('high S', CALIBRATION_WINDOW_NAME)
        lowV = cv2.getTrackbarPos('low V', CALIBRATION_WINDOW_NAME)
        highV = cv2.getTrackbarPos('high V', CALIBRATION_WINDOW_NAME)

        lower_hsv_mask = np.array([lowH, lowS, lowV])
        higher_hsv_mask = np.array([highH, highS, highV])

        hsv_mask = cv2.inRange(hsv_img, lower_hsv_mask, higher_hsv_mask)
        hsv_mask_img = cv2.bitwise_and(img, img, mask=hsv_mask)

        cv2.imshow(CALIBRATION_WINDOW_NAME, hsv_mask_img)

        if cv2.waitKey(100) == ord('q'):
            print(f'H:[{lowH}, {highH}] S:[{lowS}, {highS}] V:[{lowV}, {highV}]')
            cv2.destroyAllWindows()
            break

if should_calculate_histograms:
    # Calculate histograms for each channel respectively
    # H: 0, S: 1, V: 2
    hist_h = cv2.calcHist([hsv_img], [0], None, [MAX_H], [MIN_H, MAX_H])
    hist_s = cv2.calcHist([hsv_img], [1], None, [MAX_S], [MIN_S, MAX_S])
    hist_v = cv2.calcHist([hsv_img], [2], None, [MAX_V], [MIN_V, MAX_V])

    # Print histograms if needed
    plt.figure('Hsv histogram')
    plt.plot(hist_h, color='red')
    plt.plot(hist_s, color='green')
    plt.plot(hist_v, color='blue')
    plt.show()

    # Hard-coded threshold values of Saturation channel
    min_hsv_s, max_hsv_s = 15, 30
    lower_hsv_mask = np.array([MIN_H, min_hsv_s, MIN_V])
    higher_hsv_mask = np.array([MAX_H, max_hsv_s, MAX_V])

# Threshold image removing pixels out of range of the Saturation threshold
threshold_img = cv2.inRange(hsv_img, lower_hsv_mask, higher_hsv_mask)  # gives grayscale img with 255 and 0 values

# Canny filter method
blurred_img = cv2.GaussianBlur(threshold_img, (5, 5), 0)
canny_img = cv2.Canny(blurred_img, 100, 200)

# Subtraction method
blurred_alot_img = cv2.GaussianBlur(threshold_img, (49, 49), 0)
subtracted_img = cv2.subtract(threshold_img, blurred_alot_img)
_, subtracted_img = cv2.threshold(subtracted_img, 100, 255, cv2.THRESH_BINARY)

for c_img in canny_img, subtracted_img:
    cnts = cv2.findContours(c_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        # Compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Draw the center of the shape on the image
        cv2.circle(c_img, (cX, cY), 3, (255, 255, 255), -1)

# Plot results
plt.figure('Results')

# Plot first method
plt.subplot(2, 3, 1), plt.imshow(img, aspect='auto'), plt.title('Original image')
plt.subplot(2, 3, 2), plt.imshow(threshold_img, cmap='gray', aspect='auto'), plt.title('Grayscale HSV threshold image')
plt.subplot(2, 3, 3), plt.imshow(blurred_img, cmap='gray', aspect='auto'), plt.title('Blurred image')
plt.subplot(2, 3, 4), plt.imshow(canny_img, cmap='gray', aspect='auto'), plt.title('Edges via Canny image')
plt.subplot(2, 3, 5), plt.imshow(subtracted_img, cmap='gray', aspect='auto'), plt.title('Edges via Subtraction image')

plt.show()

# Terminate plots
plt.close()