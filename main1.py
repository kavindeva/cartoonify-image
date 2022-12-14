import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Image input
inputImage = cv.imread("images/house.jpg")
cv.imshow("original image", inputImage)

# Convert to grayscale
grayScaleImage = cv.cvtColor(inputImage, cv.COLOR_BGR2GRAY)
# cv.imshow("Grayscale image", grayScaleImage)

# applying median blur to smoothen an image
smoothGrayScale = cv.medianBlur(grayScaleImage, 7)
# cv.imshow("Smoothen image", smoothGrayScale)

# retrieving the edges for cartoon effect by using thresholding technique
getEdge = cv.adaptiveThreshold(smoothGrayScale, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 7)
# cv.imshow("Edge detection image", getEdge)


# Method will reduce the number of colors in the image, and it will create a cartoon-like effect. Color quantization
# is performed by using the K-means clustering algorithm for displaying output with a limited number of colors.
def color_quantization(image, k_value):
    # Transform the image
    data = np.float32(image).reshape((-1, 3))
    # Determine criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    # Implementing K-Means clustering algorithm
    ret, label, center = cv.kmeans(data, k_value, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result


totalColor = 9

# Reducing Colour Palette
reducedColorPalette = color_quantization(inputImage, totalColor)
# cv.imshow("Reduced Colour Palette", reducedColorPalette)

# Sometimes after blurring or smoothening an image may also tend to smooth the edges. To avoid that we will use the
# Bilateral filer
bilateralFilter = cv.bilateralFilter(inputImage, d=7, sigmaColor=200, sigmaSpace=200)
# cv.imshow("Bilateral Filter", bilateralFilter)

# Combining Edge Mask with Colored image
cartoonImage = cv.bitwise_and(bilateralFilter, bilateralFilter, mask=getEdge)
cv.imshow("Final Cartoon image", cartoonImage)

# Save cartooned image
# inputName = str(inputImage)
# print(inputName)
# fileName = "cartooned=" + inputName
# print(fileName)
cv.imwrite("cartoon-images/cartooned-image.jpg", cartoonImage)

# Plotting the whole transition
# images = [inputImage, grayScaleImage, smoothGrayScale, getEdge, reducedColorPalette, bilateralFilter, cartoonImage]

# # Define a figure of size (8, 8)
# figure = plt.figure(figsize=(20, 20))
# # Define rows and columns
# rows, cols = 2, 4
# for i in range(0, cols * rows):
#     figure.add_subplot(rows, cols, i + 1)
#     plt.imshow(images[i])
#
# print("done")
cv.waitKey(0)
