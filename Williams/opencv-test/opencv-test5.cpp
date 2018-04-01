#include <iostream>
#include <opencv2/opencv.hpp>

#define PI 3.14159265358979

using namespace cv;
using namespace std;

#define PICTURE_MODE 0

Mat pHSV;

typedef struct _Armor {
    float id;
    float x;
    float y;
    float distance;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} Armor;

typedef struct _SearchArea {
    float center;
    float size;
} SearchArea;

#ifndef NDEBUG

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
    int nTargetColor = 2;
    const int TARGET_RED = 2;
    const int TARGET_BLUE = 0;
    Scalar sTargetColor;
    if (nTargetColor == TARGET_BLUE) {
        sTargetColor = Scalar(255, 0, 0);
    } else {
        sTargetColor = Scalar(0, 0, 255);
    }
    Mat pSrcImage, pDstImage, pGrayImage, pDarkImage, pMarginImage, pResultImage, pContourEllipse, pContour, pBinaryBrightness;//, pBinaryColor;
    vector<Armor> armors;

#if PICTURE_MODE == 1
    pSrcImage = imread(argv[1], 1);
#else
    VideoCapture cap;
    if (argc == 2) {
        cap.open(argv[1]);
    } else {
        cap.open(2);
    }
    if (!cap.isOpened()) {
        printf("No image data \n");
        return -1;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CV_CAP_PROP_FPS, 30);
//    cap.set(CV_CAP_PROP_BRIGHTNESS, 0);
//    cap.set(CV_CAP_PROP_CONTRAST, 0.5);
//    cap.set(CV_CAP_PROP_SATURATION, 0.5);
//    cap.set(CV_CAP_PROP_HUE, 0.5);
//    cap.set(CV_CAP_PROP_GAIN, 0);
    cap.set(CV_CAP_PROP_EXPOSURE, 0.3);
//    cap.set(CV_CAP_PROP_XI_AUTO_WB, 0);

    cap >> pSrcImage;
#endif //if PICTURE_MODE == 1

#ifndef NDEBUG
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Contours", CV_WINDOW_AUTOSIZE);
    namedWindow("Color Points", CV_WINDOW_AUTOSIZE);
    namedWindow("Ellipses", CV_WINDOW_AUTOSIZE);
    namedWindow("Result image", WINDOW_AUTOSIZE);

    setMouseCallback("Original Image", CallBackFunc, NULL);
    setMouseCallback("Contours", CallBackFunc, NULL);
    setMouseCallback("Color Points", CallBackFunc, NULL);
    setMouseCallback("Ellipses", CallBackFunc, NULL);
    setMouseCallback("Result image", CallBackFunc, NULL);

    printf("width = %.2f\n", cap.get(CV_CAP_PROP_FRAME_WIDTH));
    printf("height = %.2f\n", cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    printf("fbs = %.2f\n", cap.get(CV_CAP_PROP_FPS));
    printf("brightness = %.2f\n", cap.get(CV_CAP_PROP_BRIGHTNESS));
    printf("contrast = %.2f\n", cap.get(CV_CAP_PROP_CONTRAST));
    printf("saturation = %.2f\n", cap.get(CV_CAP_PROP_SATURATION));
    printf("hue = %.2f\n", cap.get(CV_CAP_PROP_HUE));
    printf("exposure = %.2f\n", cap.get(CV_CAP_PROP_EXPOSURE));

    bool playVideo = true;
#endif //ifndef NDEBUG

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

    Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, 4) * 255.0);

    for (int tenFrame; pSrcImage.data; tenFrame++) {
        frameCount++;
        clock_t startTime, endTime;
        startTime = clock();
        /// make the image darker to avoid over expose
//        pSrcImage.convertTo(pDarkImage, -1, 1, -50);

        ///gamma correction
        LUT(pSrcImage, lookUpTable, pDarkImage);

#ifndef NDEBUG
        imshow("Original Image", pDarkImage);
        pDarkImage.copyTo(pResultImage);
#endif
        pContourEllipse = Mat::zeros(pSize, CV_8UC1);
        pContour = Mat::zeros(pSize, CV_8UC1);
//        pBinaryColor = Mat::zeros(pSize, CV_8UC1);
        pBinaryBrightness = Mat::zeros(pSize, CV_8UC1);
//        pResultImage = Mat::zeros(pSize, CV_8UC3);
        /// Color difference detection
        vector<Point> colorPoint;
        cvtColor(pDarkImage, pHSV, COLOR_BGR2HSV); //convert the original image into HSV colorspace

        inRange(pHSV, Scalar(0, 0, 200), Scalar(179, 200, 255), pBinaryBrightness);

//        if (nTargetColor == TARGET_RED) {
//            Mat pBinaryColorLower, pBinaryColorUpper;
//            inRange(pHSV, Scalar(0, 100, 100), Scalar(5, 255, 255), pBinaryColorLower);
//            inRange(pHSV, Scalar(175, 100, 100), Scalar(179, 255, 255), pBinaryColorUpper);
//            pBinaryColor = pBinaryColorLower | pBinaryColorUpper;
//        } else {
//            inRange(pHSV, Scalar(115, 100, 100), Scalar(125, 255, 255), pBinaryColor);
//        }

//        pBinaryBrightness = pBinaryBrightness | pBinaryColor;

#ifndef NDEBUG
        imshow("Color Points", pBinaryBrightness);
#endif
        /// edge detection
//        cvtColor(pDarkImage, pGrayImage, COLOR_RGB2GRAY);
        blur(pBinaryBrightness, pMarginImage, Size(3, 3));//use blur to reduce noise
        Canny(pMarginImage, pMarginImage, 100, 200);

        /// Margin detection and ellipse fitting
        vector<Vec4i> hierarchy;
        std::vector<std::vector<Point>> contours;
        /// Find contours
        findContours(pMarginImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
        /// Find the rotated rectangles and ellipses for each contour
        vector<RotatedRect> minEllipse;
//        unsigned int colorPointSize = colorPoint.size();
        vector<float> brightnessRatio;
        unsigned int size = contours.size();
        for (int i = 0; i < size; i++) {
            unsigned int pointsOnContour = contours[i].size();
            int red = 0, blue = 0;
            if (pointsOnContour > 5) {
                for (int j = 0; j < pointsOnContour; j++) {
                    int color = pHSV.at<Vec3b>(contours[i][j])[0];
                    if (color <= 60 || color >= 150) red++;
                    else blue++;
                }
                if ((nTargetColor == TARGET_RED && red < blue) || (nTargetColor == TARGET_BLUE && blue < red)) continue;
                RotatedRect tmp = fitEllipse(Mat(contours[i]));
                float heightToWidth = (float) tmp.size.height / tmp.size.width;
                if ((tmp.angle < 30 || tmp.angle > 150) && tmp.size.width > 3 && heightToWidth > 1.5 &&
                    heightToWidth < 9) {
                    minEllipse.push_back(tmp);
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
        int count = minEllipse.size();
        RotatedRect e1, e2;
        for (int i = 0; i < count - 1; i++) {
            for (int j = i + 1; j < count; j++) {
                e1 = minEllipse[i];
                e2 = minEllipse[j];
                float angleDifference = abs(e1.angle - e2.angle);
                float heightDifferenceRatio = abs(e1.size.height - e2.size.height) / (e1.size.height + e2.size.height);
                float widthDifferenceRatio = abs(e1.size.width - e2.size.width) / (e1.size.width + e2.size.width);
                float xDifferenceRatio = abs(e1.center.x - e2.center.x) / (e1.size.height + e2.size.height);
                float yDifferenceRatio = abs(e1.center.y - e2.center.y) / (e1.size.height + e2.size.height);
                if ((angleDifference < 5 || angleDifference > 175) && heightDifferenceRatio < 0.1 &&
                    xDifferenceRatio > 0.5 &&
                    xDifferenceRatio < 3 && yDifferenceRatio < 0.3 && widthDifferenceRatio < 0.3) {
                    Armor armor;
                    unsigned int armorsSize = armors.size();
                    if (armorsSize > 0) {
                        armor.id = armors[armorsSize - 1].id + 1;
                    } else {
                        armor.id = 0;
                    }
                    if (e1.center.x > e2.center.x) {
                        RotatedRect tmp = e1;
                        e1 = e2;
                        e2 = tmp;
                    }
                    float angle1 = e1.angle * PI / 180;
                    float angle2 = e2.angle * PI / 180;
                    float sin1 = sin(angle1), sin2 = sin(angle2), cos1 = cos(angle1), cos2 = cos(angle2);
                    Point upperLeft = Point((int) round(e1.center.x - sin1 * e1.size.height),
                                            (int) round(e1.center.y + cos1 * e1.size.height));
                    Point lowerLeft = Point((int) round(e1.center.x + sin1 * e1.size.height),
                                            (int) round(e1.center.y - cos1 * e1.size.height));
                    if (upperLeft.y > lowerLeft.y) {
                        Point tmp = upperLeft;
                        upperLeft = lowerLeft;
                        lowerLeft = tmp;
                    }
                    Point upperRight = Point((int) round(e2.center.x - sin2 * e2.size.height),
                                             (int) round(e2.center.y + cos2 * e2.size.height));
                    Point lowerRight = Point((int) round(e2.center.x + sin2 * e2.size.height),
                                             (int) round(e2.center.y - cos2 * e2.size.height));
                    if (upperRight.y > lowerRight.y) {
                        Point tmp = upperRight;
                        upperRight = lowerRight;
                        lowerRight = tmp;
                    }
                    armor.x = -(-upperRight.x * lowerRight.x * upperLeft.y + lowerRight.x * lowerLeft.x * upperLeft.y +
                                upperLeft.x * lowerLeft.x * upperRight.y - lowerRight.x * lowerLeft.x * upperRight.y +
                                upperLeft.x * upperRight.x * lowerRight.y - upperLeft.x * lowerLeft.x * lowerRight.y -
                                upperLeft.x * upperRight.x * lowerLeft.y + upperRight.x * lowerRight.x * lowerLeft.y) /
                              (upperRight.x * upperLeft.y - lowerLeft.x * upperLeft.y - upperLeft.x * upperRight.y +
                               lowerRight.x * upperRight.y - upperRight.x * lowerRight.y + lowerLeft.x * lowerRight.y +
                               upperLeft.x * lowerLeft.y - lowerRight.x * lowerLeft.y);
                    armor.y = -(-(-lowerRight.x * upperLeft.y + upperLeft.x * lowerRight.y) *
                                (upperRight.y - lowerLeft.y) + (upperLeft.y - lowerRight.y) *
                                                               (-lowerLeft.x * upperRight.y +
                                                                upperRight.x * lowerLeft.y)) /
                              ((-upperRight.x + lowerLeft.x) * (upperLeft.y - lowerRight.y) -
                               (-upperLeft.x + lowerRight.x) * (upperRight.y - lowerLeft.y));
                    armor.distance = 275.0 / (e1.size.height + e2.size.height);



                    armors.push_back(armor);
#ifndef NDEBUG
                    ellipse(pResultImage, e1, sWhite);
                    ellipse(pResultImage, e2, sWhite);
                    line(pResultImage, upperLeft, upperRight, sTargetColor, 4);
                    line(pResultImage, upperRight, lowerRight, sTargetColor, 4);
                    line(pResultImage, lowerRight, lowerLeft, sTargetColor, 4);
                    line(pResultImage, lowerLeft, upperLeft, sTargetColor, 4);
                    Point center = Point(armor.x, armor.y);
                    circle(pResultImage, center, 4, sTargetColor);

                    putText(pResultImage, to_string(armor.distance) + " m", center, FONT_HERSHEY_SIMPLEX, 1,
                            sTargetColor, 2);
#endif
                }
            }
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
        cout << (double) (endTime - startTime) / CLOCKS_PER_SEC << endl;
    }

#ifndef NDEBUG
    cout << "average time: " << (double) (clock() - totalTime) / CLOCKS_PER_SEC / frameCount << ", FPS:"
         << frameCount / (double) (clock() - totalTime) * CLOCKS_PER_SEC << endl;
#endif

#if PICTURE_MODE == 0
    cap.release();
#endif
    return 0;
}
