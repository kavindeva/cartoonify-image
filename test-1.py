import cv2 as cv


# # Read in an image
# imageRead = cv.imread('images/human-face-1.jpeg')
# # resizedImage = cv.resize(imageRead, (570, 700), interpolation=cv.INTER_AREA)
# cv.imshow('resizedImage', imageRead)

class MyImage:
    def __init__(self, img_name):
        self.img = cv.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name


x = MyImage('images/house.jpg')
print(str(x))
cv.imshow(None, x.img)


cv.waitKey(0)
cv.destroyAllWindows()
