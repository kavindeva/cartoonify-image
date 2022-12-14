from collections import defaultdict
import cv2 as cv
import numpy as np
from scipy import stats


def update_c(cin, hist1):
    while True:
        groups = defaultdict(list)
        for i in range(len(hist1)):
            if hist1[i] == 0:
                continue
            d = np.abs(cin - i)
            index = np.argmin(d)
            groups[index].append(i)
        newc = np.array(cin)
        for i, indice in groups.items():
            if np.sum(hist1[indice]) == 0:
                continue
            newc[i] = int(np.sum(indice * hist1[indice]) / np.sum(hist1[indice]))
        if np.sum(newc - cin) == 0:
            break
        cin = newc
    return cin, groups


# Calculate K-means clustering
def K_histogram(hist):
    alpha = 0.001
    nvalue = 80
    cen = np.array([128])

    while True:
        cout, groups = update_c(cen, hist)
        newcen = set()
        for i, indice in groups.items():
            if len(indice) < nvalue:
                newcen.add(cout[i])
                continue

            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                left = 0 if i == 0 else cout[i - 1]
                right = len(hist) - 1 if i == len(cout) - 1 else cout[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (cout[i] + left) / 2
                    c2 = (cout[i] + right) / 2
                    newcen.add(c1)
                    newcen.add(c2)
                else:
                    newcen.add(cout[i])
            else:
                newcen.add(cout[i])
        if len(newcen) == len(cout):
            break
        else:
            cout = np.array(sorted(newcen))
    return cout


# Module to create Cartoon
def cartoonify(image):
    # Apply Bilateral filter to our cartoon function
    kernel = np.ones((2, 2), np.uint8)
    npImage = np.array(image)
    x, y, c = npImage.shape
    print(x)
    print(y)
    print(c)

    for i in range(c):
        npImage[:, :, i] = cv.bilateralFilter(npImage[:, :, i], 5, 150, 150)
    print(npImage)

    # Canny Edge-detection
    edgeimage = cv.Canny(npImage, 100, 200)
    cv.imshow('edgedetection', edgeimage)
    output = cv.cvtColor(npImage, cv.COLOR_RGB2HSV)
    cv.imshow('cvtcolor', edgeimage)
    hists = []
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)

    centroidvalue = []
    for h in hists:
        centroidvalue.append(K_histogram(h))
    print("centroids: {0}".format(centroidvalue))

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - centroidvalue[i]), axis=1)
        output[:, i] = centroidvalue[i][index]
    output = output.reshape((x, y, c))
    output = cv.cvtColor(output, cv.COLOR_HSV2RGB)

    contours, _ = cv.findContours(edgeimage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(output, contours, -1, 0, thickness=1)
    # cartoon = cv2.bitwise_and(output, output, mask=contours)
    for i in range(3):
        output[:, :, i] = cv.erode(output[:, :, i], kernel, iterations=1)
    laplacian = cv.Laplacian(output, cv.CV_8U, ksize=11)
    output = output - laplacian

    return output


# Read in an image
imageRead = cv.imread('images/house.jpg')
# resizedImage = cv.resize(imageRead, (570, 700), interpolation=cv.INTER_AREA)
# cv.imshow('resizedImage', resizedImage)
outputImage = cartoonify(imageRead)
cv.imwrite("cartoon.jpg", outputImage)

# # cv.imshow('face', img)
# resizedImage = cv.resize(img, (570, 700), interpolation=cv.INTER_AREA)
# cv.imshow('resizedImage', resizedImage)

cv.waitKey(0)
