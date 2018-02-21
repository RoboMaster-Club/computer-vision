#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ARGS_MODE 0
#define PICTURE_MODE 0

int main(int argc, char **argv) {
#if PICTURE_MODE == 1
#if ARGS_MODE == 1
    pSrc = imread(argv[1], 1);
#else
    pSrc = imread("test1.png", 1);
#endif
#else
    bool playVideo = true;
#if ARGS_MODE == 1
    VideoCapture cap(argv[1]);
#else
    VideoCapture cap("../../RedCar.avi");
#endif
    if (!cap.isOpened()) {
        printf("No image data \n");
        return -1;
    }
    clock_t startTime, endTime;
    Mat pSrc, pGary, pBinaryBrightness, pHSV, pBinaryColor, pEnemyColor;
    cap >> pSrc;
    Size imgSize = pSrc.size();
#endif
    namedWindow("test", WINDOW_AUTOSIZE);
    Scalar targetColor = Scalar(0, 0, 255);
    while (pSrc.data) {
        startTime = clock();
        cvtColor(pSrc, pGary, COLOR_RGB2GRAY);//convert the original image into gray image
        threshold(pGary, pBinaryBrightness, 200, 255, THRESH_BINARY);//convert gray image into binary brightness image
        cvtColor(pSrc, pHSV, COLOR_BGR2HSV); //convert the original image into HSV colorspace
        if (targetColor == Scalar(0, 0, 255)) {
            Mat pBinaryColorLower, pBinaryColorUpper;
            inRange(pHSV, Scalar(0, 50, 50), Scalar(10, 255, 255), pBinaryColorLower);
            inRange(pHSV, Scalar(160, 50, 50), Scalar(179, 255, 255), pBinaryColorUpper);
            pBinaryColor = pBinaryColorLower + pBinaryColorUpper;
        } else {
            inRange(pHSV, Scalar(75, 100, 100), Scalar(135, 255, 255), pBinaryColor);
        }

        pEnemyColor = pBinaryBrightness & pBinaryColor;
        int erodeSize = 5;
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * erodeSize + 1, 2 * erodeSize + 1), Point(erodeSize, erodeSize));
        dilate(pEnemyColor, pEnemyColor, element);

        imshow("test", pEnemyColor);

#ifndef NDEBUG
        /// Press  ESC on keyboard to  exit
        char c = (char) waitKey(1);
        if (c == 27)
            break;
#if PICTURE_MODE == 0
        else if (c == ' ')
            playVideo = !playVideo;
        if (playVideo)
#endif
#endif
            cap >> pSrc;
        endTime = clock();
        cout << (double) (endTime - startTime) / CLOCKS_PER_SEC << endl;
    }

#if PICTURE_MODE == 0
    cap.release();
#endif
    return 0;
}