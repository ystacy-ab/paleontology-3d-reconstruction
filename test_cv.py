import cv2
import numpy as np

image = cv2.imread('images/image3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

threshold_value = 130
_, thresh_simple = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

_, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 15, 5)

cv2.imshow('Original', image)
cv2.imshow('Simple Threshold', thresh_simple)
cv2.imshow('Otsu Threshold', thresh_otsu)
cv2.imshow('Adaptive Threshold', thresh_adaptive)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('masks_code/fossil_simple.png', thresh_simple)
cv2.imwrite('masks_code/fossil_otsu.png', thresh_otsu)
cv2.imwrite('masks_code/fossil_adaptive.png', thresh_adaptive)