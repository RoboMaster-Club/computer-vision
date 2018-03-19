import sys
import cv2
import numpy as np
import copy
import math
import time

CAP_WIDTH = 1920
CAP_HEIGHT = 1080

def rect_is_vertical(rect):
    if rect[1][0] > rect[1][1]:
        if rect[2] <= -45.0:
            return 1

    if rect[1][0] < rect[1][1]:
        if rect[2] >= -45.0:
            return 1
    return 0


def rect_size_cmp(rect_a, rect_b, ratio):
    area_a = rect_a[1][0] * rect_a[1][1]
    area_b = rect_b[1][0] * rect_b[1][1]
    if ratio * area_a < area_b <= area_a:
        return 1
    if ratio * area_b < area_a <= area_b:
        return 1

    return 0


def rect_dist(rect_a, rect_b):
    dx = rect_a[0][0] - rect_b[0][0]
    dy = rect_a[0][1] - rect_b[0][1]

    return (dx ** 2 + dy ** 2) ** (1 / 2)


def rect_length(rect):
    if rect[1][0] > rect[1][1]:
        return rect[1][0]
    return rect[1][1]


def rect_area(rect):
    return rect[1][0] * rect[1][1]


def rect_angle_diff(rect_a, rect_b):
    angle_a = rect_a[2]
    angle_b = rect_b[2]
    if not rect_a[1][0] < rect_a[1][1]:
        angle_a += 90
    if not rect_b[1][0] < rect_b[1][1]:
        angle_b += 90

    dangle = (angle_a - angle_b) % 180
    if dangle > 90:
        dangle = 180 - dangle
    return dangle


def draw_box(rect, img, color, thickness):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, color, thickness);


def minn(a, b):
    if a < b:
        return a
    return b


def maxx(a, b):
    if a > b:
        return a
    return b


def hori_angle(vec):
    mul = -1
    if vec[1] > 0:
        mul = 1
    return mul * math.acos(vec[0] / (vec[0] ** 2 + vec[1] ** 2) ** (1/2)) * 180 / math.pi


def find_light_bar(frame, search_range):
    #print("range",search_range)

    # find region of interest
    ser_frame = frame[search_range[0][1]:search_range[1][1], search_range[0][0]:search_range[1][0]]

    # get greyscale information
    img_gray = cv2.cvtColor(ser_frame, cv2.COLOR_RGB2GRAY)
    img_blue = cv2.split(ser_frame)[0]
    img_red = cv2.split(ser_frame)[2]
    #cv2.imshow("capture", ser_frame)

    # apply Gaussian blur
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_blue = cv2.GaussianBlur(img_blue, (5, 5), 0)
    img_red = cv2.GaussianBlur(img_red, (5, 5), 0)

    # get and merge binary imgs
    ret, binary_gray = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    ret, binary_blue = cv2.threshold(img_blue, 230, 255, cv2.THRESH_BINARY)
    ret, binary_red = cv2.threshold(img_red, 200, 255, cv2.THRESH_BINARY)

    binary = np.bitwise_or(binary_gray, binary_blue)
    binary = np.bitwise_or(binary, binary_red)

    #cv2.imshow("binary",binary)
    # find contours
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initialize rect_list
    rect_list = []

    # find all vertical rects
    for cnt in contours:
        try:
            rect = cv2.minAreaRect(cnt)
            cnt_area = cv2.contourArea(cnt)
            box_area = rect[1][0] * rect[1][1]
            #draw_box(rect, ser_frame, (255, 0, 255), 1)
            #print("area", box_area, cnt_area)
            if rect_is_vertical(rect) and cnt_area / box_area > 0.45:
                rect_list.append(rect)
                #draw_box(rect, ser_frame, (0, 255, 255), 2)
        except:
            pass

    # cv2.imshow('ser_frame',ser_frame)
    # we need at least 2 to lock on the armor plate
    if len(rect_list) < 2:
        raise Exception("not enough rectangle found")

    # initialization
    max_rect_area = 0
    am_bar = [-1, -1]

    # this is only for debugging purpose
    #print("-------------------------------------------------------")
    #frame_size = ser_frame.copy()
    #frame_len = ser_frame.copy()
    #frame_bk3 = ser_frame.copy()
    #frame_bk4 = ser_frame.copy()
    #frame_angle = ser_frame.copy()
    #frame_pos = ser_frame.copy()
    #frame_test = ser_frame.copy()

    # compare each pair and see if they meet the requirements
    for i in range(len(rect_list)):
        for j in range(i + 1, len(rect_list)):
            rect_i = rect_list[i]
            rect_j = rect_list[j]
            if not rect_size_cmp(rect_i, rect_j, 0.1):
                #print("i", rect_i[1][0] * rect_i[1][1])
                #print("j", rect_j[1][0] * rect_j[1][1])
                #print("BAKA!")
                #draw_box(rect_i, frame_size, (255, 0, 255), 1)
                #draw_box(rect_j, frame_size, (255, 0, 255), 1)
                continue
            avglen = (rect_length(rect_i) + rect_length(rect_j)) / 2

            if not (0.5 * avglen < rect_dist(rect_i, rect_j) < 6 * avglen):
                #print("BAKA!!")
                #print(rect_i)
                #print(rect_j)
                #print(0.5 * avglen, "<", rect_dist(rect_i, rect_j), "<", 4.6 * avglen)
                #draw_box(rect_i, frame_len, (255, 0, 255), 1)
                #draw_box(rect_j, frame_len, (255, 0, 255), 1)
                continue
            if not (0.7 * avglen < rect_length(rect_i) < 1.3 * avglen):
                #print("BAKA!!!")
                #draw_box(rect_i, frame_bk3, (255, 0, 255), 1)
                #draw_box(rect_j, frame_bk3, (255, 0, 255), 1)
                continue
            if not (0.7 * avglen < rect_length(rect_j) < 1.3 * avglen):
                #print("BAKA!!!!")
                #draw_box(rect_i, frame_bk4, (255, 0, 255), 1)
                #draw_box(rect_j, frame_bk4, (255, 0, 255), 1)
                continue
            if not (-45 < rect_angle_diff(rect_i, rect_j) < 45):
                #print("BAKA!!!!!")
                #print(rect_i[2], rect_j[2])
                #print(rect_angle_diff(rect_i, rect_j))
                #draw_box(rect_i, frame_angle, (255, 0, 255), 1)
                #draw_box(rect_j, frame_angle, (255, 0, 255), 1)
                continue

            #mid_seg = [rect_i[0],rect_j[0]]
            #lf = (int(rect_i[0][0]), int(rect_i[0][1]))
            #rb = (int(rect_j[0][0]), int(rect_j[0][1]))

            vec = (rect_j[0][0] - rect_i[0][0], rect_j[0][1] - rect_i[0][1])
            if vec[0] < 0:
                vec = (-1 * vec[0], -1 * vec[1])

            angle = -1 * hori_angle(vec)
            #cv2.line(frame_test, lf, rb, (255, 0, 0), 1)
            #cv2.imshow('test', frame_test)
            diff_i = angle + rect_i[2]
            if rect_i[1][0] > rect_i[1][1]:
                diff_i += 90
            diff_j = angle + rect_j[2]
            if rect_j[1][0] > rect_j[1][1]:
                diff_j += 90
            #print("di %lf, dj %lf, angle %lf"%(rect_i[2], rect_j[2],angle))
            #print("ANGLE!!!", diff_i, diff_j)
            #cv2.waitKey(0)
            if not (abs(diff_i) < 20 and abs(diff_j) < 20):
                #print("SB!")
                #draw_box(rect_i, frame_pos, (255, 0, 255), 1)
                #draw_box(rect_j, frame_pos, (255, 0, 255), 1)
                continue


            area = rect_area(rect_i) + rect_area(rect_j)
            if area > max_rect_area:
                max_rect_area = area
                am_bar[0] = rect_i
                am_bar[1] = rect_j

    #cv2.imshow('size',frame_size)
    #cv2.imshow('len',frame_len)
    #cv2.imshow('bk3',frame_bk3)
    #cv2.imshow('bk4',frame_bk4)
    #cv2.imshow('angle',frame_angle)
    #cv2.imshow('pos',frame_pos)

    # if no pair meets all requirements
    if am_bar[0] == -1 or am_bar[1] == -1:
        raise Exception("no pairs found")
    return am_bar


start = time.clock()
cap = cv2.VideoCapture("N1.mp4")
# initialize search range
ret, frame = cap.read()
CAP_WIDTH = frame.shape[1]
CAP_HEIGHT = frame.shape[0]
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20, (CAP_WIDTH, CAP_HEIGHT))
search_size = [[0, 0], [CAP_WIDTH, CAP_HEIGHT]]
while 1:
    ret, frame = cap.read()
    #cv2.imshow("frame", frame)
    #cv2.waitKey(0)
    if not ret:
        break

    try:
        # try to find light_bar
        am_bar = find_light_bar(frame, search_size)
    except Exception:

        # if doesn't find enlarger search range for the next frame
        search_len = search_size[1][0] - search_size[0][0]
        search_hig = search_size[1][1] - search_size[0][1]
        lh = search_size[0]
        rb = search_size[1]
        lh[0] = maxx(lh[0] - int(0.3 * search_len), 0)
        lh[1] = maxx(lh[1] - int(0.3 * search_hig), 0)
        rb[0] = minn(rb[0] + int(0.3 * search_len), CAP_WIDTH)
        rb[1] = minn(rb[1] + int(0.3 * search_hig), CAP_HEIGHT)

        #cv2.waitKey(0)
        search_size = [lh, rb]
        print("less rect founded than needed")
        #cv2.imshow("contours", frame)
        out.write(frame)
        continue

    #sys.stdout.flush()
    #sys.stderr.flush()

    # add coordinate shift to the light bar box
    am_bar[0] = list(am_bar[0])
    am_bar[1] = list(am_bar[1])

    am_bar[0][0] = (am_bar[0][0][0] + search_size[0][0], am_bar[0][0][1] + search_size[0][1])
    am_bar[1][0] = (am_bar[1][0][0] + search_size[0][0], am_bar[1][0][1] + search_size[0][1])

    am_bar[0] = tuple(am_bar[0])
    am_bar[1] = tuple(am_bar[1])

    # draw box and aim position
    tot_len = int(rect_length(am_bar[0]) + rect_length(am_bar[1]))
    dist = rect_dist(am_bar[0], am_bar[1])
    draw_box(am_bar[0], frame, (0, 0, 255), 2)
    draw_box(am_bar[1], frame, (255, 0, 0), 2)

    center = (int((am_bar[0][0][0] + am_bar[1][0][0]) / 2), int((am_bar[0][0][1] + am_bar[1][0][1]) / 2))
    cv2.circle(frame, center, 5, (255, 0, 255), 2)

    # update search range
    search_size[0] = [maxx(center[0] - int(2.5 * tot_len), 0), maxx(center[1] - int(2.5 * tot_len), 0)]
    search_size[1] = [minn(center[0] + int(2.5 * tot_len), CAP_WIDTH), minn(center[1] + int(2.5 * tot_len), CAP_HEIGHT)]

    #cv2.circle(frame, tuple(search_size[0]), 5, (0, 255, 255), 2)
    #cv2.circle(frame, tuple(search_size[1]), 5, (255, 255, 0), 2)
    cv2.rectangle(frame, tuple(search_size[0]), tuple(search_size[1]), (255, 255, 0), 4)

    #cv2.waitKey(0)
    #cv2.imshow("contours", frame)
    out.write(frame)
    #if cv2.waitKey(50) & 0xFF == ord('q'):
    #    break

print('Time used:', time.clock() - start)
cap.release()
cv2.destroyAllWindows()
