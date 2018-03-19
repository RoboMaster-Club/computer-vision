# for the contour drawing algorithm I've used Carlo's
import cv2
import numpy as np
from colorProcessing import *

cap = cv2.VideoCapture('RedCar.avi')
if not cap.isOpened():
    print('Error Opening Video')
while cap.isOpened():
    ret, frame = cap.read()
    # Our operations on the frame come here
    filterFrame = filterColor(frame)
    gray = cv2.cvtColor(filterFrame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh_img = cv2.threshold(blur, 91, 255, cv2.THRESH_BINARY)
    contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    # for c in contours:
    #     cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
    for c in contours:
        try:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))

            shape, approx = detect(c)

            if shape == 'lightbar':
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv2.putText(frame, 'lightbar', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # draw bisecting lines
                (x, y, w, h) = cv2.boundingRect(approx)
                cv2.line(frame, (int((x + (w / 2))), y), (int((x + (w / 2))), y + h), (255, 0, 0), 2)


        except:
            pass

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()