import cv2
import numpy as np
import imutils
cv2.__version__

def detect(c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
 
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
 
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
 
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
 
        # return the name of the shape
        return shape


cap = cv2.VideoCapture('RedCar.avi')


if not cap.isOpened():
    print('Error opening video stream or file')

while cap.isOpened():

    ret, frame = cap.read()
    
    if ret:
        
        ratio = 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([0,0,170])
        upper = np.array([190,15,255])
        mask = cv2.inRange(hsv, lower, upper)
        
        reds = cv2.bitwise_and(hsv, hsv, mask=mask)
        reds_bgr = cv2.cvtColor(reds, cv2.COLOR_HSV2BGR)
        
        gray = cv2.cvtColor(reds_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)[1]
        
             
        
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        try:
            for c in cnts:
                # compute the center of the contour, then detect the name of the
                # shape using only the contour
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                shape = detect(c)

                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                
                if shape == 'rectangle':
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                    cv2.putText(frame, 'lightbar', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        except:
            pass
        
        
        cv2.imshow('Frame', thresh)
        cv2.imshow('orig', frame)
        
    if (cv2.waitKey(25) & 0xFF == ord('q')) or not ret:
        break
        
cap.release()
cv2.destroyAllWindows()