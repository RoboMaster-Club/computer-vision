import cv2
import numpy as np
import imutils
from detect_shapes import *
print('Using opencv version: ' + cv2.__version__)

cap = cv2.VideoCapture('RedCar.avi')

if not cap.isOpened():
    print('Error opening video stream or file')

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([0,0,251])
        upper = np.array([180,3,255])
        mask = cv2.inRange(hsv, lower, upper)
        
        lights = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        lights_bgr = cv2.cvtColor(lights, cv2.COLOR_HSV2BGR)
        blurred = cv2.GaussianBlur(lights_bgr, (5,5), 0)
        
        thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]
        
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        
        
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        prev_coords = None
        
        
        for c in cnts:
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

                    #draw bisecting lines
                    (x, y, w, h) = cv2.boundingRect(approx)
                    cv2.line(frame, (int((x+(w/2))), y), (int((x+(w/2))), y+h), (255,0,0), 2)


            except:
                pass
        
        
        
        cv2.imshow('Output', frame)
        
    if (cv2.waitKey(25) & 0xFF == ord('q')) or not ret:
        break
        
cap.release()
cv2.destroyAllWindows()