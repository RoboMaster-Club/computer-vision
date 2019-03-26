import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time

image_path = os.path.join(os.getcwd(), 'images\\')

#in BGR format
blue_lower = np.array([150, 0, 0])
blue_upper = np.array([255, 100, 100])
red_lower = np.array([0, 0, 150])
red_upper = np.array([100, 100, 255])

def pixel_in_range(pixel, lower, upper):
    if  (pixel[0] >= lower[0] and pixel[0] <= upper[0] and
        pixel[1] >= lower[1] and pixel[1] <= upper[1] and
        pixel[2] >= lower[2] and pixel[2] <= upper[2]):
        return True
    else:
        return False

#brute force
def method1(image, x1, y1, x2, y2):
    start = time.time()

    armor = image[y1:y2, x1:x2]
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

    print(blue_count, "out of", w*h)
    print(red_count, "out of", w*h)
    print(time.time() - start, "seconds")
    return "blue" if blue_count > red_count else "red"
    

def main():
    blue_image = cv2.imread(image_path + "robot_blue_1m_480p_frame0.jpg")
    red_image = cv2.imread(image_path + "robot_red_1m_480p_frame244.jpg")

    """ Show red and blue image information """
    # plt.imshow(blue_image)
    # plt.show()
    # plt.imshow(red_image)
    # plt.show()

    """ Test red and blue upper and lower bounds """
    # armor = blue_image[114:233, 337:472] #y1:y2, x1:x2
    # cv2.imshow('armor', armor)
    # mask = cv2.inRange(armor, blue_lower, blue_upper)
    # cv2.imshow('blue mask', mask)
    # mask = cv2.inRange(armor, red_lower, red_upper)
    # cv2.imshow('red mask', mask)

    """ Test methods """
    print(method1(blue_image, 377, 114, 472, 233))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()