import cv2
import numpy as np

imagesPath = "./images"
img = cv2.imread("./images/image_2.jpg")
blurredImg = cv2.GaussianBlur(img, (5, 5), 0)
hsvImg = cv2.cvtColor(blurredImg, cv2.COLOR_BGR2HSV)

def shapeDetection(cnt):
    arcLength = cv2.arcLength(cnt, True)
    approxPolygon = cv2.approxPolyDP(cnt, 0.01*arcLength, True)
    numOfVertices = len(approxPolygon)
    if numOfVertices == 3:
        shape = "Triangle"
    elif numOfVertices == 4:
        shape = "Rectangle"
    elif numOfVertices == 5:
        shape = 'Pentagon'
    else:
        shape = 'Circle'
    return shape, approxPolygon

# Range for color yellow
lowerBound = np.array([20, 122, 100])
higherBound = np.array([40, 255, 255])
mask = cv2.inRange(hsvImg, lowerBound, higherBound)
yellow = cv2.bitwise_and(hsvImg, hsvImg, mask=mask)

threshedHsv = cv2.threshold(yellow, 60, 255, cv2.THRESH_BINARY)[1]
threshedBGR = cv2.cvtColor(threshedHsv, cv2.COLOR_HSV2BGR)
grayThreshed = cv2.cvtColor(threshedBGR, cv2.COLOR_BGR2GRAY)
originImg, contours, hieracrchy = cv2.findContours(grayThreshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find and plot contours and bounding Box
for contour in contours:
    shape, approxPolygon = shapeDetection(contour)
    img = cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(approxPolygon)
    print(shape)
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv2.imshow('Final', img)
cv2.waitKey()

cv2.destroyAllWindows()
