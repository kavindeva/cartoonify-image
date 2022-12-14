import os
import sys
import cv2 as cv
import easygui
import numpy as np
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt

top = tk.Tk()
top.geometry('400x400')
top.title('Cartoonify Your Image !')
top.configure(background='white')
label = Label(top, background='#CDCDCD', font=('calibri', 20, 'bold'))


def upload():
    ImagePath = easygui.fileopenbox()
    cartoonify(ImagePath)
    # print(ImagePath)


def cartoonify(ImagePath):
    inputimage = cv.imread(ImagePath)

    # confirm that image is chosen
    if inputimage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()
    height, width, channels = inputimage.shape
    print("Height: {}".format(height))
    print("width: {}".format(width))

    # Image resize to fit the display screen
    def imageresize(imageinput):
        if height < width:
            print("It's a Landscape image")
            if height > 1300 or width > 700:
                resizedimage = cv.resize(imageinput, (1200, 700), cv.INTER_AREA)  # (width, height)
                inputimage1 = resizedimage
            else:
                inputimage1 = imageinput
        else:
            print("It's a Portrait Image")
            if height > 1300 or width > 700:
                resizedimage = cv.resize(imageinput, (600, 700), cv.INTER_AREA)  # (width, height)
                inputimage1 = resizedimage
            else:
                inputimage1 = imageinput
        return inputimage1

    resizedimage1 = imageresize(inputimage)
    cv.imshow("original image", resizedimage1)
    # imagename = imagename[7:]
    # print(imagename)

    # Convert to grayscale
    grayscaleimage = cv.cvtColor(resizedimage1, cv.COLOR_BGR2GRAY)
    # cv.imshow("Grayscale image", grayscaleimage)

    # applying median blur to smoothen an image
    smoothgrayscale = cv.medianBlur(grayscaleimage, 7)
    # cv.imshow("Smoothen image", smoothgrayscale)

    # retrieving the edges for cartoon effect by using thresholding technique
    getEdge = cv.adaptiveThreshold(smoothgrayscale, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 7)
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
    reducedColorPalette = color_quantization(resizedimage1, totalColor)
    # cv.imshow("Reduced Colour Palette", reducedColorPalette)

    # Sometimes after blurring or smoothening an image may also tend to smooth the edges. To avoid that we will use the
    # Bilateral filer
    bilateralFilter = cv.bilateralFilter(resizedimage1, d=7, sigmaColor=200, sigmaSpace=200)
    # cv.imshow("Bilateral Filter", bilateralFilter)

    # Combining Edge Mask with Colored image
    cartoonImage = cv.bitwise_and(bilateralFilter, bilateralFilter, mask=getEdge)
    cv.imshow("Final Cartoon image", cartoonImage)

    # Plotting the whole transition
    # images = [resizedimage1, grayscaleimage, smoothgrayscale, getEdge, reducedColorPalette, bilateralFilter,
    #           cartoonImage]

    save1 = Button(top, text="Save cartoon image", command=lambda: save(cartoonImage, ImagePath), padx=30, pady=5)
    save1.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
    save1.pack(side=TOP, pady=50)

    plt.show()


def save(ReSized6, savepath):
    # saving an image using imwrite()
    newName = "cartooned-"
    path1 = "C:\\Users\\kavin\\Documents\\cartoonify-image\\cartoon-images"
    # print(path1)
    extension = os.path.splitext(savepath)[1]
    oldname = os.path.splitext(savepath)[0]
    oldname = oldname[49:]
    # print(extension)
    # print(oldname)
    path = os.path.join(path1, newName + oldname + extension)
    # print(path)
    cv.imwrite(path, ReSized6)
    i = "Image saved by name " + newName + " at " + path
    tk.messagebox.showinfo(title=None, message=i)


upload = Button(top, text="Cartoonify an Image", command=upload, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
upload.pack(side=TOP, pady=50)

top.mainloop()

# print("done")
# cv.waitKey(0)
