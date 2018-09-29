"""
script to detect the yellow object in 'image_2.jpg'
"""

import cv2
import numpy as np

def isolateElement(src, mask):
    """
    isolate a element/object of a image
    :param src: the original image in HSV format
    :param mask: the range of HSV colors to isolate
    :return: a image in BGR format
    """

    #isolate element
    object = cv2.bitwise_and(src, src, mask=mask)
    cv2.imshow('After Masking', object)

    #blur
    objectBGR = cv2.cvtColor(object, cv2.COLOR_HSV2BGR)
    objectBlur = cv2.GaussianBlur(objectBGR, (5, 5), 0, 0)
    cv2.imshow('After Blur', objectBlur)

    #thresholding
    _, objectThresh = cv2.threshold(objectBlur, 60, 255, cv2.THRESH_BINARY)
    cv2.imshow('After Thresholding', objectBlur)

    return objectThresh

def detectShape(contour):
    """
    find the shape of a contour
    :param contour: a single contour from findContours()
    :return numSides: the number of sides of the shape
    :return vertices: the vertices of the shape
    """

    arcLength = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.06*arcLength, True)

    return len(vertices), vertices

def main():
    img = cv2.imread('image_2.jpg')
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # isolate the yellow object
    lower = np.array([20, 180, 180])
    higher = np.array([30, 255, 255])
    mask = cv2.inRange(imgHSV, lower, higher)
    object = isolateElement(imgHSV, mask)

    #detect shape
    objectGrey = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)
    _, objectContours, _ = cv2.findContours(objectGrey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for countour in objectContours:
        numSides, verticies = detectShape(countour)

        if numSides >= 4: #if its a circle
            cv2.drawContours(img, [countour], -1, (0, 0, 0), 1)

            #draw rectangle around the object
            (x, y, w, h) = cv2.boundingRect(verticies)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 1)


    cv2.imshow('After Object Detection', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()