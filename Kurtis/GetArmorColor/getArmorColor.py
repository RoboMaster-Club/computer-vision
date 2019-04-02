import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time
import math

image_path = os.path.join(os.getcwd(), 'images\\')

#in BGR format
blue_lower = np.array([130, 0, 0])
blue_upper = np.array([255, 80, 80])
red_lower = np.array([0, 0, 110])
red_upper = np.array([100, 100, 255])

""" helper functions """
def pixel_in_range(pixel, lower, upper):
    if  (pixel[0] >= lower[0] and pixel[0] <= upper[0]
    and pixel[1] >= lower[1] and pixel[1] <= upper[1]
    and pixel[2] >= lower[2] and pixel[2] <= upper[2]):
        return True
    else:
        return False

""" get armor color method solutions """
def pixel_count(armor):
    start = time.time()

    h = armor.shape[0]
    w = armor.shape[1]

    blue_count = 0
    for x in range(0, w):
        for y in range(0, h):
            if pixel_in_range(armor[y, x], blue_lower, blue_upper):
                blue_count += 1

    red_count = 0
    for x in range(0, w):
        for y in range(0, h):
            if pixel_in_range(armor[y, x], red_lower, red_upper):
                red_count += 1

    # print("blue count:", blue_count, "out of", w*h)
    # print("red_count:", red_count, "out of", w*h)
    print(time.time() - start, "seconds")
    return "blue" if blue_count > red_count else "red"

def blob_count(armor):
    start = time.time()

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.inRange(armor, blue_lower, blue_upper)
    #cv2.imshow('blue mask org', blue_mask)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=3)
    red_mask = cv2.inRange(armor, red_lower, red_upper)
    #cv2.imshow('red mask org', red_mask)
    red_mask = cv2.dilate(red_mask, kernel, iterations=3)

    #cv2.imshow('red mask', red_mask)
    #cv2.imshow('blue mask', blue_mask)

    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_count = 0 if blue_contours is None else len(blue_contours)
    red_count = 0 if red_contours is None else len(red_contours)

    #cv2.drawContours(armor, red_contours, -1, (0, 255, 0), 1)
    #cv2.drawContours(armor, blue_contours, -1, (0, 255, 0), 1)
    #armor = cv2.resize(armor, (400, 400))
    #cv2.imshow('armor', armor)

    #print("blue count:", blue_count)
    print("red_count:", red_count)
    #print(time.time() - start, "seconds")

    if (blue_count > red_count):
        return "blue"
    elif (red_count > blue_count):
        return "red"
    elif (red_count == 2 and blue_count == 2):
        x0, _, _, _ = cv2.boundingRect(blue_contours[0])
        x1, _, _, _ = cv2.boundingRect(blue_contours[1])
        blue_x = min(x1, x0)

        x0, _, _, _ = cv2.boundingRect(red_contours[0])
        x1, _, _, _ = cv2.boundingRect(red_contours[1])
        red_x = min(x1, x0)
        return "blue" if blue_x > red_x else "red" #don't choose the contour that is closest to the edge of the image
    else:
        return "None"

def main():
    blue_image = cv2.imread(image_path + "robot_blue_1m_480p_frame0.jpg")
    blue_armor = blue_image[114:233, 330:482]
    red_image = cv2.imread(image_path + "robot_red_1m_480p_frame244.jpg")
    red_armor = red_image[100:300, 300:470]
    edge_image = cv2.imread(image_path + "robot_blue_1m_480p_edge_case.jpg")
    edge_armor = edge_image[114:250, 330:515]

    """ Edge case creation """
    # red_mask = cv2.inRange(red_armor, red_lower, red_upper)
    # red_contours, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # i = 0
    # contour_max = 0
    # area_max = 0
    # for contour in red_contours: #assume there is contours
    #     if area_max < cv2.contourArea(contour):
    #         contour_max = i
    #         area_max = cv2.contourArea(contour)
    #     i += 1

    # x,y,w,h = cv2.boundingRect(red_contours[contour_max])
    # red_blob = red_armor[y:(y+h), x:(x+w)]
    # red_blob = cv2.resize(red_blob, (30, 120))

    # cv2.imshow('red', red_blob)

    # x, y = 480, 125
    # blue_image[y:(120+y), x:(30+x)] = red_blob
    # cv2.imshow('blue', blue_image)
    # cv2.imwrite("C:\\Users\\Kurti\\RobomasterProjects\\images\\robot_blue_1m_480p_edge_case.jpg", blue_image)

    """ Show red and blue image information """
    # plt.imshow(blue_image)
    # plt.show()
    # plt.imshow(red_image)
    # plt.show()

    """ Test red and blue upper and lower bounds """
    # armor = blue_image[114:233, 330:482] #y1:y2, x1:x2
    # cv2.imshow('armor', armor)
    # mask = cv2.inRange(armor, blue_lower, blue_upper)
    # cv2.imshow('blue mask', mask)
    # mask = cv2.inRange(armor, red_lower, red_upper)
    # cv2.imshow('red mask', mask)

    """ Test methods """
    #h = 119, w = 105
    # print('pixel count:')
    # print(pixel_count(edge_armor))

    print('blob count:')
    print(blob_count(edge_armor))

    # print('left right blob count:')
    # print(left_right_blob_count(edge_armor))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()