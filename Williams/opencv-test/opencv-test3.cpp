#include <iostream>
#include <opencv2/opencv.hpp>
#include <libsvm/svm.h>


#define PI 3.14159265358979

using namespace cv;
using namespace std;

#define PICTURE_MODE 0

map<int, vector<int>> solveEllipseForX(RotatedRect rrEllipse) {
    map<int, vector<int>> result;
    float a = rrEllipse.size.width / 2.0;
    float b = rrEllipse.size.height / 2.0;
    float x0 = rrEllipse.center.x;
    float y0 = rrEllipse.center.y;
    float y02 = pow(y0, 2);
    float theta = rrEllipse.angle * PI / 180;
    float sina = sin(theta), cosa = cos(theta);
    float a2 = pow(a, 2);
    float b2 = pow(b, 2);
    float sin2a = pow(sina, 2);
    float cos2a = pow(cosa, 2);
    float sin4a = pow(sina, 4);
    float cos4a = pow(cosa, 4);
    for (int y = (int) round(y0) - 1; y > 0; y--) {
        int y2 = pow(y, 2);
        float alternativeTerm = sqrt(a2 * b2 *
                                     (a2 * sin2a + b2 * cos2a - y02 * sin4a - y02 * cos4a - 2 * y02 * sin2a * cos2a +
                                      2 * y0 * y * sin4a + 2 * y0 * y * cos4a + 4 * y0 * y * sin2a * cos2a -
                                      y2 * sin4a - y2 * cos4a - 2 * y2 * sin2a * cos2a));
        float staticTerm = a2 * x0 * sin2a - a2 * y0 * sina * cosa + a2 * y * sina * cosa + b2 * x0 * cos2a +
                           b2 * y0 * sina * cosa - b2 * y * sina * cosa;
        float denominator = a2 * sin2a + b2 * cos2a;
        float x1 = (staticTerm - alternativeTerm) / denominator;
        float x2 = (staticTerm + alternativeTerm) / denominator;
        if (!isnan(x1) && !isnan(x2)) {
            result[y] = {(int) round(x1), (int) round(x2)};
        } else {
            break;
        }
    }
    for (int y = (int) round(y0);; y++) {
        int y2 = pow(y, 2);
        float alternativeTerm = sqrt(a2 * b2 *
                                     (a2 * sin2a + b2 * cos2a - y02 * sin4a - y02 * cos4a - 2 * y02 * sin2a * cos2a +
                                      2 * y0 * y * sin4a + 2 * y0 * y * cos4a + 4 * y0 * y * sin2a * cos2a -
                                      y2 * sin4a - y2 * cos4a - 2 * y2 * sin2a * cos2a));
        float staticTerm = a2 * x0 * sin2a - a2 * y0 * sina * cosa + a2 * y * sina * cosa + b2 * x0 * cos2a +
                           b2 * y0 * sina * cosa - b2 * y * sina * cosa;
        float denominator = a2 * sin2a + b2 * cos2a;
        float x1 = (-alternativeTerm + staticTerm) / denominator;
        float x2 = (alternativeTerm + staticTerm) / denominator;

        if (!isnan(x1) && !isnan(x2)) {
            result[y] = {(int) round(x1), (int) round(x2)};
        } else {
            break;
        }
    }

    return result;
}

/*
bool pointComp(Point p1, Point p2) {
    if (p1.y < p2.y) {
        return true;
    } else if (p1.y > p2.y) {
        return false;
    } else {
        if (p1.x < p2.x) {
            return true;
        } else {
            return false;
        }
    }
}
 */

bool ellipseComp(RotatedRect e1, RotatedRect e2) {
    if (e1.center.y < e2.center.y) {
        return true;
    } else if (e1.center.y > e2.center.y) {
        return false;
    } else {
        if (e1.center.x < e2.center.x) {
            return true;
        } else {
            return false;
        }
    }
}

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

void
parallelCalculation(vector<RotatedRect> minEllipse, vector<Point> colorPoint, int start, int end, int colorPointSize,
                    vector<float> *ellipseBrightnessRatio) {
    for (int i = start; i < end; i++) {
        int count = 0;
        for (int j = 0; j < colorPointSize; j++) {
            if (isPointInEllipse(minEllipse[i], colorPoint[j])) {
//                circle(pPointInEllipse, colorPoint[j], 1, sTargetColor);
                count++;
            }
        }
        ellipseBrightnessRatio->push_back(count / (PI * minEllipse[i].size.height * minEllipse[i].size.width) * 4);
    }
}

#ifndef NDEBUG
Mat pHSV;

void CallBackFunc(int event, int x, int y, int flags, void *userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point p = Point(x, y);
        cout << pHSV.at<Vec3b>(p) << endl;
    }
}
#endif

int main(int argc, char **argv) {

    ///variables
    Scalar sWhite = Scalar(255, 255, 255);
    Scalar sTargetColor = Scalar(0, 0, 255);
    int nTargetColor = 2;
    Mat pSrcImage, pDstImage, pGrayImage, pDarkImage, pMarginImage, pResultImage, pContourEllipse, pContour;
#ifdef NDEBUG
    Mat pHSV;
#endif


#ifndef NDEBUG
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Contours", CV_WINDOW_AUTOSIZE);
    namedWindow("Ellipses", CV_WINDOW_AUTOSIZE);
//    namedWindow("Point in ellipses", WINDOW_AUTOSIZE);
    namedWindow("Result image", WINDOW_AUTOSIZE);
    setMouseCallback("Ellipses", CallBackFunc, NULL);
    bool playVideo = true;
#endif //ifndef NDEBUG

#if PICTURE_MODE == 1
    pSrcImage = imread(argv[1], 1);
#else
    VideoCapture cap;
    if (argc == 2) {
        cap.open(argv[1]);
    } else {
        cap.open(0);
    }
    if (!cap.isOpened()) {
        printf("No image data \n");
        return -1;
    }

    cap >> pSrcImage;
#endif //if PICTURE_MODE == 1

    Size pSize = pSrcImage.size();
    int type = pSrcImage.type();
    int height = pSrcImage.rows;
    int width = pSrcImage.cols;

    ///initialize images
    pDstImage.create(pSize, type);
    pDarkImage.create(pSize, type);
    pResultImage.create(pSize, type);
    pContour.create(pSize, type);

    clock_t totalTime = clock();
    long int frameCount = 0;

    while (pSrcImage.data) {
        frameCount++;
        clock_t startTime, endTime;
        startTime = clock();
        /// make the image darker to avoid over expose
        pSrcImage.convertTo(pDarkImage, -1, 1, -50);
#ifndef NDEBUG
        imshow("Original Image", pSrcImage);
#endif
        /// Picture contours + light points in ellipses
        pContourEllipse = Mat::zeros(pSize, CV_8UC1);
        pContour = Mat::zeros(pSize, CV_8UC1);
        /// Color difference detection
        vector<Point> colorPoint;
//        pPointImage = Mat::zeros(pSize, CV_8UC1);
        cvtColor(pSrcImage, pHSV, COLOR_BGR2HSV); //convert the original image into HSV colorspace
        if (sTargetColor == Scalar(0, 0, 255)) {
//            Mat pBinaryColorLower, pBinaryColorUpper;
//            inRange(pHSV, Scalar(0, 0, 200), Scalar(40, 100, 256), pBinaryColorLower);
//            inRange(pHSV, Scalar(170, 0, 200), Scalar(180, 100, 256), pBinaryColorUpper);
//            pContourEllipse = pBinaryColorLower | pBinaryColorUpper;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    Point p(j, i);
                    Vec3b hsv = pHSV.at<Vec3b>(p);
                    if (((hsv[0] >= 0 && hsv[0] <= 40) || (hsv[0] >= 140 && hsv[0] <= 179)) && hsv[1] >= 0 &&
                        hsv[1] <= 100 && hsv[2] >= 200 && hsv[2] <= 255) {
                        colorPoint.push_back(p);
                    }
                }
            }
        } else {
//            inRange(pHSV, Scalar(75, 50, 50), Scalar(135, 255, 255), pContourEllipse);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    Point p(j, i);
                    Vec3b hsv = pHSV.at<Vec3b>(p);
                    if (hsv[0] >= 50 && hsv[0] <= 70 && hsv[1] >= 0 && hsv[1] <= 100 && hsv[2] >= 200 &&
                        hsv[2] <= 255) {
                        colorPoint.push_back(p);
                    }
                }
            }
        }

        /// edge detection
        cvtColor(pDarkImage, pGrayImage, COLOR_RGB2GRAY);
        blur(pGrayImage, pMarginImage, Size(3, 3));//use blur to reduce noise
        Canny(pMarginImage, pMarginImage, 100, 100 * 2);

        /// Margin detection and ellipse fitting
        vector<Vec4i> hierarchy;
        std::vector<std::vector<Point>> contours;
        /// Detect edges using Threshold
//        threshold(pMarginImage, pMarginImage, 200, 255, THRESH_BINARY);
        /// Find contours
        findContours(pMarginImage, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));

        /// Find the rotated rectangles and ellipses for each contour
        vector<RotatedRect> minEllipse;
        unsigned int colorPointSize = colorPoint.size();
        vector<float> brightnessRatio;
        unsigned int size = contours.size();
        for (int i = 0; i < size; i++) {
            if (contours[i].size() > 5) {
                RotatedRect tmp = fitEllipse(Mat(contours[i]));
                if ((tmp.angle < 30 || tmp.angle > 150) && tmp.size.height > 3 && tmp.size.width > 3) {
                    int count = 0;
                    for (int j = 0; j < colorPointSize; j++) {
                        if (isPointInEllipse(tmp, colorPoint[j])) {
                            count++;
                        }
                    }
                    float ratio = count / (PI * tmp.size.height * tmp.size.width) * 4;
                    if (ratio > 0.5) {
                        brightnessRatio.push_back(ratio);
                        minEllipse.push_back(tmp);
                    }
                }
            }
        }
#ifndef NDEBUG
        for (int i = 0; i < contours.size(); i++) {
            drawContours(pContour, contours, i, sWhite);
        }
        imshow("Contours", pContour);
        size = minEllipse.size();
        for (int i = 0; i < size; i++) {
            ellipse(pContourEllipse, minEllipse[i], sWhite, 1, 8);
            putText(pContourEllipse, to_string(i), minEllipse[i].center, FONT_HERSHEY_SIMPLEX, 1, sWhite);
        }
        for (int i = 0; i < colorPoint.size(); i++) {
            circle(pContourEllipse, colorPoint[i], 1, sWhite);
        }
        imshow("Ellipses", pContourEllipse);
#endif

        /// Find the shield
        std::sort(minEllipse.begin(), minEllipse.end(), ellipseComp);
        multimap<float, int> ellipseRank;
        size = brightnessRatio.size();
        for (int i = 0; i < size; i++) {
            ellipseRank.insert(pair<float, int>(brightnessRatio[i], i));
        }

        pResultImage = Mat::zeros(pSrcImage.size(), CV_8UC3);
        pSrcImage.copyTo(pResultImage);
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
//                if ((it->second == 19 && it2->second == 20) || (it->second == 20 && it2->second == 19)) {
//                    cout << "";
//                }
                if ((angleDifference < 10 || angleDifference > 170) && heightDifferenceRatio < 0.1 &&
                    xDifferenceRatio > 0.5 &&
                    xDifferenceRatio < 3 && yDifferenceRatio < 0.5 && widthDifferenceRatio < 0.3) {
                    ellipse(pResultImage, e1, sWhite);
                    ellipse(pResultImage, e2, sWhite);
                    stopFlag = true;
#ifndef NDEBUG
                    if (playVideo)
#endif
//                        cout << "1 1:" << e1.size.height + e2.size.height << " 2:" << e1.center.y + e2.center.y << " 3:"
//                             << abs(e1.size.height - e2.size.height)
//                             << " 4:" << abs(e1.center.x - e2.center.x) << endl;
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

            line(pResultImage, p1.y < p2.y ? p1 : p2, p1.y > p2.y ? p1 : p2, sTargetColor, 4);
            line(pResultImage, p1.y > p2.y ? p1 : p2, p3.y > p4.y ? p3 : p4, sTargetColor, 4);
            line(pResultImage, p3.y > p4.y ? p3 : p4, p3.y < p4.y ? p3 : p4, sTargetColor, 4);
            line(pResultImage, p3.y < p4.y ? p3 : p4, p1.y < p2.y ? p1 : p2, sTargetColor, 4);
            float distance = 60 / (e1.size.height + e2.size.height);

            putText(pResultImage, to_string(distance) + " m", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1,
                    sTargetColor, 2);
        }
#ifndef NDEBUG
        imshow("Result image", pResultImage);
#if PICTURE_MODE == 0
        /// Press  ESC on keyboard to  exit
        char c = (char) waitKey(1);
        if (c == 27)
            break;
        else if (c == ' ')
            playVideo = !playVideo;
        if (playVideo)
            cap >> pSrcImage;
#else
        waitKey(0);
        break;
#endif
#else
        cap >> pSrcImage;
#endif //ifndef NDEBUG

        endTime = clock();
//        cout << (double) (endTime - startTime) / CLOCKS_PER_SEC << endl << endl;
    }

    cout << "average time: " << (double)(clock() - totalTime) / CLOCKS_PER_SEC / frameCount << endl;

#if PICTURE_MODE == 0
    cap.release();
#endif
    return 0;
}
