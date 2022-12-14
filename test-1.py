import cv2 as cv

# Read in an image
imageRead = cv.imread('images/human-face-1.jpeg')
# resizedImage = cv.resize(imageRead, (570, 700), interpolation=cv.INTER_AREA)
cv.imshow('resizedImage', imageRead)

cv.waitKey(0)
cv.destroyAllWindows()
