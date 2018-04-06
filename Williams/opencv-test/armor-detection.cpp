#include <iostream>
#include <opencv2/opencv.hpp>

#define PI 3.14159265358979

using namespace cv;
using namespace std;

typedef struct _Armor {
    int id;
    float width;
    float height;
    float x;
    float y;
    float z; //distance
    float angular_velocity_x;
    float angular_velocity_y;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} Armor;

typedef struct _SearchArea {
    int id;
    Rect rect;
} SearchArea;

#define PICTURE_MODE 0

//Mat pHSV;
Mat lookUpTable(1, 256, CV_8U);
int nTargetColor = 2;
const int TARGET_RED = 2;
const int TARGET_BLUE = 0;
int height = 0;
int width = 0;

#ifndef NDEBUG
const Scalar sWhite = Scalar(255, 255, 255);
Scalar sTargetColor;

void CallBackFunc(int event, int x, int y, int flags, void *userdata) {
//    if (event == EVENT_LBUTTONDOWN) {
//        Point p = Point(x, y);
//        cout << pHSV.at<Vec3b>(p) << endl;
//    }
}

#endif


bool detect(const Mat &pSrcImage, const SearchArea &curSearchArea, const Armor *referenceArmor, vector<Armor> &armors) {
    Mat pDarkImage, pMarginImage, pBinaryBrightness, pHSV;//, pBinaryColor;
    Size pSize = curSearchArea.rect.size();
    vector<Armor> resultArmors;

    ///gamma correction
    LUT(pSrcImage, lookUpTable, pDarkImage);
    pBinaryBrightness = Mat::zeros(pSize, CV_8UC1);

    /// Color difference detection
    vector<Point> colorPoint;
    cvtColor(pDarkImage, pHSV, COLOR_BGR2HSV); //convert the original image into HSV colorspace
    inRange(pHSV, Scalar(0, 0, 200), Scalar(179, 200, 255), pBinaryBrightness);

    /// edge detection
    blur(pBinaryBrightness, pMarginImage, Size(3, 3));//use blur to reduce noise
    Canny(pMarginImage, pMarginImage, 100, 200);

    /// Margin detection and ellipse fitting
    vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> contours;
    /// Find contours
    findContours(pMarginImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
    vector<RotatedRect> minEllipse;
//        unsigned int colorPointSize = colorPoint.size();
    /// Fit & filter ellipses
    unsigned int contourSize = (unsigned int) contours.size();
    for (int i = 0; i < contourSize; i++) {
        unsigned int pointsOnContour = (unsigned int) contours[i].size();
        if (pointsOnContour > 5) {
            int red = 0, blue = 0;
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

    /// Match ellipses to form armors
    int count = (int) minEllipse.size();
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
                Armor tmpArmor;
                unsigned int armorsSize = armors.size();
                if (armorsSize > 0) {
                    tmpArmor.id = armors[armorsSize - 1].id + 1;
                } else {
                    tmpArmor.id = 0;
                }
                tmpArmor.id = curSearchArea.id;
                if (e1.center.x > e2.center.x) {
                    RotatedRect tmp = e1;
                    e1 = e2;
                    e2 = tmp;
                }
                tmpArmor.width = e2.center.x - e1.center.x;
                tmpArmor.height = e1.size.height + e2.size.height;
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
                tmpArmor.x = curSearchArea.rect.tl().x -
                             (-upperRight.x * lowerRight.x * upperLeft.y + lowerRight.x * lowerLeft.x * upperLeft.y +
                              upperLeft.x * lowerLeft.x * upperRight.y - lowerRight.x * lowerLeft.x * upperRight.y +
                              upperLeft.x * upperRight.x * lowerRight.y - upperLeft.x * lowerLeft.x * lowerRight.y -
                              upperLeft.x * upperRight.x * lowerLeft.y + upperRight.x * lowerRight.x * lowerLeft.y) /
                             (upperRight.x * upperLeft.y - lowerLeft.x * upperLeft.y - upperLeft.x * upperRight.y +
                              lowerRight.x * upperRight.y - upperRight.x * lowerRight.y + lowerLeft.x * lowerRight.y +
                              upperLeft.x * lowerLeft.y - lowerRight.x * lowerLeft.y);
                tmpArmor.y = curSearchArea.rect.tl().y - (-(-lowerRight.x * upperLeft.y + upperLeft.x * lowerRight.y) *
                                                          (upperRight.y - lowerLeft.y) + (upperLeft.y - lowerRight.y) *
                                                                                         (-lowerLeft.x * upperRight.y +
                                                                                          upperRight.x * lowerLeft.y)) /
                                                         ((-upperRight.x + lowerLeft.x) * (upperLeft.y - lowerRight.y) -
                                                          (-upperLeft.x + lowerRight.x) * (upperRight.y - lowerLeft.y));
                tmpArmor.z = (float) 275 / (e1.size.height + e2.size.height);

                if (referenceArmor != NULL) {
                    tmpArmor.angular_velocity_x = tmpArmor.x - referenceArmor->x;
                    tmpArmor.angular_velocity_y = tmpArmor.y - referenceArmor->y;
                    tmpArmor.velocity_z = (tmpArmor.z - referenceArmor->z);
                } else {
                    tmpArmor.angular_velocity_y = 0;
                    tmpArmor.angular_velocity_x = 0;
                    tmpArmor.velocity_z = 0;
                }
                resultArmors.push_back(tmpArmor);
            }
        }
    }
    if (resultArmors.empty()) {
        return false;
    }
    armors = resultArmors;
    return true;
}

bool detect(const Mat &pSrcImage, const SearchArea &curSearchArea, const Armor &referenceArmor, Armor &armor) {
    vector<Armor> armors;
    bool found = detect(pSrcImage, curSearchArea, &referenceArmor, armors);
    if (!found) return false;
    unsigned int armorsCount = (unsigned int) armors.size();
    if (armorsCount > 1) {
        float minDifference =
                abs(armors[0].height - referenceArmor.height) + abs(armors[0].width - referenceArmor.width) +
                abs(armors[0].x - referenceArmor.x) + abs(armors[0].y - referenceArmor.y) +
                abs(armors[0].z - referenceArmor.z);
        int index = 0;
        for (int i = 1; i < armorsCount; i++) {
            float difference =
                    abs(armors[i].height - referenceArmor.height) +
                    abs(armors[i].width - referenceArmor.width) +
                    abs(armors[i].x - referenceArmor.x) + abs(armors[i].y - referenceArmor.y) +
                    abs(armors[i].z - referenceArmor.z);
            if (difference < minDifference) {
                minDifference = difference;
                index = i;
            }
        }
        armor = armors[index];
        armor.id = curSearchArea.id;
    } else {
        armor = armors[0];
        armor.id = curSearchArea.id;
    }
    return true;
}

void getSearchArea(const vector<Armor> &armors, vector<SearchArea> &searchAreas) {
    unsigned int size = armors.size();
    for (int i = 0; i < size; i++) {
        Armor curArmor = armors[i];
        searchAreas[i].id = armors[i].id;
        float x = curArmor.x - curArmor.width * 3 / 2 + curArmor.angular_velocity_x;
        x = x > 0 ? x : 0;
        float y = curArmor.y - curArmor.height * 3 / 2 + curArmor.angular_velocity_y;
        y = y > 0 ? y : 0;
        float saWidth = curArmor.width * 3;
        saWidth = x + saWidth > width ? width - x : saWidth;
        float saHeight = curArmor.height * 3;
        saHeight = y + saHeight > height ? height - y : saHeight;
        searchAreas[i].rect = Rect(x, y, saWidth, saHeight);
    }
}

int main(int argc, char **argv) {

    Mat pSrcImage;

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

    /// Camera setup
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

    Mat pResultImage;

    if (nTargetColor == TARGET_BLUE) {
        sTargetColor = Scalar(255, 0, 0);
    } else {
        sTargetColor = Scalar(0, 0, 255);
    }

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Result image", WINDOW_AUTOSIZE);

    setMouseCallback("Original Image", CallBackFunc, NULL);
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

    vector<Armor> armors;
    vector<SearchArea> searchAreas;
    Size pSize = pSrcImage.size();
    int type = pSrcImage.type();
    height = pSrcImage.rows;
    width = pSrcImage.cols;

    clock_t totalTime = clock();
    long int frameCount = 0;

    uchar *p = lookUpTable.ptr();

    SearchArea frame;
    frame.id = 0;
    frame.rect = Rect(0, 0, width, height);

    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, 4) * 255.0);

    for (int tenFrame = 0; pSrcImage.data; tenFrame++) {
        frameCount++;
        clock_t startTime, endTime;
        startTime = clock();

        int searchAreaCount = searchAreas.size();
        if (searchAreaCount == 0) {
            detect(pSrcImage, frame, NULL, armors);
        } else {
            // TODO - Multi-threading
            for (int i = 0; i < searchAreaCount; i++) {
                Mat subImage(pSrcImage, searchAreas[i].rect);
                bool found = detect(subImage, searchAreas[i], armors[i], armors[i]);
                if (!found) {
                    armors.erase(armors.begin() + i);
                    searchAreas.erase(searchAreas.begin() + i);
                    searchAreaCount--;
                    i--;
                }
            }
            if (tenFrame == 11) {
                ///cover all the known armors with black rect
                for (int i = 0; i < searchAreaCount; i++) {
                    rectangle(pSrcImage, Rect(armors[i].x - armors[i].width / 2, armors[i].y - armors[i].height / 2,
                                              armors[i].width, armors[i].height), Scalar(0, 0, 0), CV_FILLED);
                }
                vector<Armor> newArmors;
                bool found = detect(pSrcImage, frame, NULL, newArmors);
                if (found)
                    armors.insert(armors.end(), newArmors.begin(), newArmors.end());
                tenFrame = 0;
            }
        }
        searchAreas.resize(armors.size());
        getSearchArea(armors, searchAreas);


#ifndef NDEBUG
        pSrcImage.copyTo(pResultImage);
        for (int i = 0; i < armors.size(); i++) {
            circle(pResultImage, Point(armors[i].x, armors[i].y), 5, sTargetColor, CV_FILLED);
            rectangle(pResultImage, searchAreas[i].rect, sTargetColor, 1);
            putText(pResultImage, to_string(armors[i].z) + " m", Point(armors[i].x, armors[i].y), FONT_HERSHEY_SIMPLEX,
                    1, sTargetColor, 2);
            putText(pResultImage, "vx: " + to_string(armors[i].angular_velocity_x) + " p/f",
                    Point(armors[i].x, armors[i].y + 25), FONT_HERSHEY_SIMPLEX,
                    1, sTargetColor, 2);
            putText(pResultImage, "vy: " + to_string(armors[i].angular_velocity_y) + " p/f",
                    Point(armors[i].x, armors[i].y + 50), FONT_HERSHEY_SIMPLEX,
                    1, sTargetColor, 2);
            putText(pResultImage, "vz: " + to_string(armors[i].velocity_z) + " m/s",
                    Point(armors[i].x, armors[i].y + 75),
                    FONT_HERSHEY_SIMPLEX,
                    1, sTargetColor, 2);
        }
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
    cout << "average time: " << (double) (clock() - totalTime) / CLOCKS_PER_SEC / frameCount << ", FPS:"
         << frameCount / (double) (clock() - totalTime) * CLOCKS_PER_SEC << endl;

#if PICTURE_MODE == 0
    cap.release();
#endif
    return 0;
}
