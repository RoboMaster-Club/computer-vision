#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ARGS_MODE 1
#define PICTURE_MODE 0
#define PI 3.14159265358979323846

bool isPointInEllipse(RotatedRect ellipse, Point p) {
    if (p.x < ellipse.center.x - ellipse.size.height || p.x > ellipse.center.x + ellipse.size.height ||
        p.y < ellipse.center.y - ellipse.size.height || p.y > ellipse.center.y + ellipse.size.height) {
        return false;
    }
    float alpha = ellipse.angle * PI / 180;
    float cosa = cos(alpha), sina = sin(alpha);
    return pow((cosa * (p.x - ellipse.center.x) + sina * (p.y - ellipse.center.y)) / ellipse.size.width * 2,
               2) +
           pow((sina * (p.x - ellipse.center.x) - cosa * (p.y - ellipse.center.y)) / ellipse.size.height * 2,
               2) <= 1;
}


int main(int argc, char **argv) {
    clock_t startTime, endTime;
    Mat pSrc, pGary, pHSV, pBinaryColor, pMarginImage, pContours, pEnemyColor, pPointInEllipse, pResult;
    ///initialize images

#if PICTURE_MODE == 1
#if ARGS_MODE == 1
    pSrcImage = imread(argv[1], 1);
#else
    pSrcImage = imread("../../test1.png", 1);
#endif
#else
#if ARGS_MODE == 1
    VideoCapture cap(argv[1]);
#else
    //    VideoCapture cap("../../RedCar.avi");
        VideoCapture cap(0);
#endif
    if (!cap.isOpened()) {
        printf("No image data \n");
        return -1;
    }

    cap >> pSrc;
#endif

    Size imgSize = pSrc.size();
    int type = pSrc.type();
    unsigned int height = pSrc.rows;
    unsigned int width = pSrc.cols;
#ifndef NDEBUG
    namedWindow("Result image", WINDOW_AUTOSIZE);
    namedWindow("Binary color", WINDOW_AUTOSIZE);
    namedWindow("Contours", WINDOW_AUTOSIZE);
    namedWindow("Src", WINDOW_AUTOSIZE);
    bool playVideo = true;
#endif
    Scalar targetColor = Scalar(0, 0, 255);
    while (pSrc.data) {
#ifndef NDEBUG
        imshow("Src", pSrc);
#endif
        startTime = clock();
        cvtColor(pSrc, pGary, COLOR_RGB2GRAY);//convert the original image into gray image
        //threshold(pGary, pBinaryBrightness, 200, 255, THRESH_BINARY);//convert gray image into binary brightness image
        vector<Point> brightPoint;
        for (int i = 0; i < imgSize.height; i++) {
            for (int j = 0; j < imgSize.width; j++) {
                brightPoint.push_back(Point(j, i));
            }
        }
        cvtColor(pSrc, pHSV, COLOR_BGR2HSV); //convert the original image into HSV colorspace
        if (targetColor == Scalar(0, 0, 255)) {
            Mat pBinaryColorLower, pBinaryColorUpper;
            inRange(pHSV, Scalar(0, 50, 50), Scalar(10, 255, 255), pBinaryColorLower);
            inRange(pHSV, Scalar(160, 50, 50), Scalar(179, 255, 255), pBinaryColorUpper);
            pBinaryColor = pBinaryColorLower | pBinaryColorUpper;
        } else {
            inRange(pHSV, Scalar(75, 100, 100), Scalar(135, 255, 255), pBinaryColor);
        }

#ifndef NDEBUG
        imshow("Binary color", pBinaryColor);
#endif

        vector<Vec4i> hierarchy;
        vector<vector<Point> > contours;

//        pEnemyColor = pBinaryBrightness & pBinaryColor;
        blur(pBinaryColor, pBinaryColor, Size(3, 3));
        Canny(pBinaryColor, pMarginImage, 100, 200, 3);
#ifndef NDEBUG
        imshow("Contours", pMarginImage);
        findContours(pMarginImage, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
#endif
        /*
        vector<RotatedRect> minEllipse;
        unsigned int contourCount = contours.size();
        for (int i = 0; i < contourCount; i++) {
            if (contours[i].size() > 5)
                minEllipse.push_back(fitEllipse(Mat(contours[i])));
        }

        unsigned int ellipseCount = minEllipse.size();
        unsigned int brightPointCount = brightPoint.size();

//        map<float, vector<RotatedRect>> ellipseRank;
        vector<float> brightnessRatio(ellipseCount);

        for (int i = 0; i < ellipseCount; i++) {
            int count = 0;
            for (int j = 0; j < brightPointCount; j++) {
                if (isPointInEllipse(minEllipse[i], brightPoint[j])) {
//                    circle(pPointInEllipse, brightPoint[j], 1, brightPoint);
                    count++;
                }
            }
//            float ratio = count / (PI * minEllipse[i].size.height * minEllipse[i].size.width) * 4;
//            if(ellipseRank.at(ratio).empty()){
//                ellipseRank.insert(pair<float,vector<RotatedRect>>(ratio, vector<RotatedRect>(1, minEllipse[i])));
//            } else {
//                ellipseRank.at(ratio).push_back(minEllipse[i]);'
            brightnessRatio[i] = count / (PI * minEllipse[i].size.height * minEllipse[i].size.width) * 4;
        }

        multimap<float, int> ellipseRank;
        pResult = Mat::zeros(imgSize, CV_8UC3);
        pSrc.copyTo(pResult);
        RotatedRect e1;
        RotatedRect e2;
        bool stopFlag = false;
        for (multimap<float, int>::reverse_iterator it = ellipseRank.rbegin(); it != ellipseRank.rend(); it++) {
            for (multimap<float, int>::reverse_iterator it2 = next(ellipseRank.rbegin());
                 it2 != ellipseRank.rend(); it2++) {
                e1 = minEllipse[it->second];
                e2 = minEllipse[it2->second];
                float angleDifference = abs(e1.angle - e2.angle);
                float heightDifferenceRatio = abs(e1.size.height - e2.size.height) / (e1.size.height + e2.size.height);
                float widthDifferenceRatio = abs(e1.size.width - e2.size.width) / (e1.size.width + e2.size.width);
                float xDifferenceRatio = abs(e1.center.x - e2.center.x) / (e1.size.height + e2.size.height);
                float yDifferenceRatio = abs(e1.center.y - e2.center.y) / (e1.size.height + e2.size.height);
//                if ((it->second == 0 && it2->second == 2) || (it->second == 2 && it2->second == 0)) {
//                    cout << "";
//                }
                if ((angleDifference < 7 || angleDifference > 173) && heightDifferenceRatio < 0.1 &&
                    xDifferenceRatio > 0.5 &&
                    xDifferenceRatio < 2.5 && yDifferenceRatio < 0.2 && widthDifferenceRatio < 0.3) {
                    ellipse(pResult, e1, Scalar(255, 255, 255));
                    ellipse(pResult, e2, Scalar(255, 255, 255));
                    stopFlag = true;
                    break;
                }
            }
            if (stopFlag) break;
        }

        if (stopFlag) {

            float angle1 = e1.angle * PI / 180;
            float angle2 = e2.angle * PI / 180;

            Point p1((int) round(e1.center.x - sin(angle1) * e1.size.height),
                     (int) round(e1.center.y + cos(angle1) * e1.size.height));
            Point p2((int) round(e1.center.x + sin(angle1) * e1.size.height),
                     (int) round(e1.center.y - cos(angle1) * e1.size.height));
            Point p3((int) round(e2.center.x - sin(angle2) * e2.size.height),
                     (int) round(e2.center.y + cos(angle2) * e2.size.height));
            Point p4((int) round(e2.center.x + sin(angle2) * e2.size.height),
                     (int) round(e2.center.y - cos(angle2) * e2.size.height));

            line(pResult, p1.y < p2.y ? p1 : p2, p1.y > p2.y ? p1 : p2, targetColor, 4);
            line(pResult, p1.y > p2.y ? p1 : p2, p3.y > p4.y ? p3 : p4, targetColor, 4);
            line(pResult, p3.y > p4.y ? p3 : p4, p3.y < p4.y ? p3 : p4, targetColor, 4);
            line(pResult, p3.y < p4.y ? p3 : p4, p1.y < p2.y ? p1 : p2, targetColor, 4);
            float distance = 70 / (e1.size.height + e2.size.height);

            putText(pResult, to_string(distance) + " m", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1,
                    targetColor, 2);


        }
         */


#ifndef NDEBUG
        //imshow("Result image", pResult);

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
