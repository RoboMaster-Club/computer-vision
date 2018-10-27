#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <inRange_gpu.h>

#include "Settings.h"
#include "SearchArea.h"
#include "Armor.h"
#include "I2C.h"

using namespace cv;
using namespace std;

//#define FBF //frame by frame

int nTargetColor = 2;
const int TARGET_RED = 2;
const int TARGET_BLUE = 0;

#ifndef NDEBUG
const Scalar sWhite = Scalar(255, 255, 255);
Scalar sTargetColor;

//void CallBackFunc(int event, int x, int y, int flags, void *userdata) {
//    if (event == EVENT_LBUTTONDOWN) {
//        Point p = Point(x, y);
//        cout << ((Mat *) userdata)->at<Vec3b>(p) << endl;
//    }
//}

#endif

bool detect(const Mat &mSrcImage, const Rect &curSearchArea, const Armor *referenceArmor, vector<Armor> &armors,
            Ptr<cuda::LookUpTable> lookupTable, const float xCoefficient, const float yCoefficient,
            const float zCoefficient) {

    armors.clear();

    Size size = mSrcImage.size();
    int type = mSrcImage.type();

    cuda::GpuMat pSrcImage, pDarkImage(size, type), pHSV(size, type), pBinaryColor(size, CV_8UC1), pBinaryBrightness(
            size, CV_8UC1);

    pSrcImage.upload(mSrcImage);


    ///gamma correction
    lookupTable->transform(pSrcImage, pDarkImage);

    /// Color difference detection
    vector<Point> colorPoint;
    cuda::cvtColor(pDarkImage, pHSV, COLOR_BGR2HSV); //convert the original image into HSV colorspace

#ifdef FBF
    //    imshow("dark image", pDarkImage);
#endif

    inRange_gpu(pHSV, Scalar(0, 0, 200), Scalar(179, 200, 255), pBinaryBrightness);

    if (nTargetColor == TARGET_RED) {
        cuda::GpuMat pBinaryColorLower(size, CV_8UC1), pBinaryColorUpper(size, CV_8UC1);
        inRange_gpu(pHSV, Scalar(0, 0, 200), Scalar(30, 255, 255), pBinaryColorLower);
        inRange_gpu(pHSV, Scalar(170, 0, 200), Scalar(179, 255, 255), pBinaryColorUpper);
        cuda::bitwise_or(pBinaryColorLower, pBinaryColorUpper, pBinaryColor);
    } else {
        inRange_gpu(pHSV, Scalar(85, 0, 200), Scalar(125, 255, 255), pBinaryColor);
    }
    cuda::bitwise_or(pBinaryColor, pBinaryBrightness, pBinaryColor);

    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(100, 300, 3, false);
    canny->detect(pBinaryColor, pBinaryColor);

    Mat mHSV(size, type), mBinaryColor(size, CV_8UC1);
    pBinaryColor.download(mBinaryColor);
    pHSV.download(mHSV);

    vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> contours;
    findContours(mBinaryColor, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    vector<RotatedRect> minEllipse;

    /// Fit & filter ellipses
    auto contourSize = (unsigned int) contours.size();
    for (unsigned int i = 0; i < contourSize; i++) {
        auto pointsOnContour = (unsigned int) contours[i].size();
        if (pointsOnContour > 5) {
            int red = 0, blue = 0;
            for (unsigned int j = 0; j < pointsOnContour; j++) {
                int color = mHSV.at<Vec3b>(contours[i][j])[0];
                if (color <= 60 || color >= 150) red++;
                else blue++;
            }
            if ((nTargetColor == TARGET_RED && red < blue) || (nTargetColor == TARGET_BLUE && blue < red)) continue;
            RotatedRect tmp = fitEllipse(Mat(contours[i]));
            float heightToWidth = tmp.size.height / tmp.size.width;
            if ((tmp.angle < 30 || tmp.angle > 150) && tmp.size.width > 3 && heightToWidth > 1 &&
                heightToWidth < 10) {
                minEllipse.push_back(tmp);
#ifdef FBF
                ellipse(pBinaryBrightness, tmp, Scalar(255, 255, 255), 1, 8);
#endif
            }
        }
    }

#ifdef FBF
    imshow("binary color", mBinaryColor);
    imshow("HSV", mHSV);
//    imshow("ellipse", pBinaryBrightness);
#endif
    /// Match ellipses to form armors
    auto minEllipseSize = (int) minEllipse.size();
    RotatedRect *e1, *e2;
    for (int i = 0; i < minEllipseSize - 1; i++) {
        for (int j = i + 1; j < minEllipseSize; j++) {
            e1 = &minEllipse[i];
            e2 = &minEllipse[j];
            float angleDifference = abs(e1->angle - e2->angle);
            float heightDifferenceRatio = abs(e1->size.height - e2->size.height) / (e1->size.height + e2->size.height);
            float widthDifferenceRatio = abs(e1->size.width - e2->size.width) / (e1->size.width + e2->size.width);
            float xDifferenceRatio = abs(e1->center.x - e2->center.x) / (e1->size.height + e2->size.height);
            float yDifferenceRatio = abs(e1->center.y - e2->center.y) / (e1->size.height + e2->size.height);
            if ((angleDifference < 5 || angleDifference > 175) && heightDifferenceRatio < 0.2 &&
                xDifferenceRatio > 0.5 &&
                xDifferenceRatio < 3 && yDifferenceRatio < 0.3 && widthDifferenceRatio < 0.6) {
                Armor tmpArmor;
                if (e1->center.x > e2->center.x) {
                    RotatedRect *tmp = e1;
                    e1 = e2;
                    e2 = tmp;
                }
                tmpArmor.width = e2->center.x - e1->center.x;
                tmpArmor.height = e1->size.height + e2->size.height;
                float angle1 = e1->angle * (float) M_PI / 180;
                float angle2 = e2->angle * (float) M_PI / 180;
                float sin1 = sin(angle1), sin2 = sin(angle2), cos1 = cos(angle1), cos2 = cos(angle2);
                Point upperLeft = Point((int) round(e1->center.x - sin1 * e1->size.height),
                                        (int) round(e1->center.y + cos1 * e1->size.height));
                Point lowerLeft = Point((int) round(e1->center.x + sin1 * e1->size.height),
                                        (int) round(e1->center.y - cos1 * e1->size.height));
                if (upperLeft.y > lowerLeft.y) {
                    Point tmp = upperLeft;
                    upperLeft = lowerLeft;
                    lowerLeft = tmp;
                }
                Point upperRight = Point((int) round(e2->center.x - sin2 * e2->size.height),
                                         (int) round(e2->center.y + cos2 * e2->size.height));
                Point lowerRight = Point((int) round(e2->center.x + sin2 * e2->size.height),
                                         (int) round(e2->center.y - cos2 * e2->size.height));
                if (upperRight.y > lowerRight.y) {
                    Point tmp = upperRight;
                    upperRight = lowerRight;
                    lowerRight = tmp;
                }
                tmpArmor.x = curSearchArea.tl().x -
                             (float) (-upperRight.x * lowerRight.x * upperLeft.y +
                                      lowerRight.x * lowerLeft.x * upperLeft.y +
                                      upperLeft.x * lowerLeft.x * upperRight.y -
                                      lowerRight.x * lowerLeft.x * upperRight.y +
                                      upperLeft.x * upperRight.x * lowerRight.y -
                                      upperLeft.x * lowerLeft.x * lowerRight.y -
                                      upperLeft.x * upperRight.x * lowerLeft.y +
                                      upperRight.x * lowerRight.x * lowerLeft.y) /
                             (upperRight.x * upperLeft.y - lowerLeft.x * upperLeft.y - upperLeft.x * upperRight.y +
                              lowerRight.x * upperRight.y - upperRight.x * lowerRight.y + lowerLeft.x * lowerRight.y +
                              upperLeft.x * lowerLeft.y - lowerRight.x * lowerLeft.y);
                tmpArmor.y = curSearchArea.tl().y -
                             (float) (-(-lowerRight.x * upperLeft.y + upperLeft.x * lowerRight.y) *
                                      (upperRight.y - lowerLeft.y) + (upperLeft.y - lowerRight.y) *
                                                                     (-lowerLeft.x * upperRight.y +
                                                                      upperRight.x * lowerLeft.y)) /
                             ((-upperRight.x + lowerLeft.x) * (upperLeft.y - lowerRight.y) -
                              (-upperLeft.x + lowerRight.x) * (upperRight.y - lowerLeft.y));
                tmpArmor.z = zCoefficient / (e1->size.height + e2->size.height);

                if (referenceArmor != nullptr) {
                    tmpArmor.internal_velocity_x = tmpArmor.x - referenceArmor->x;
                    tmpArmor.internal_velocity_y = tmpArmor.y - referenceArmor->y;
                    tmpArmor.velocity_z = tmpArmor.z - referenceArmor->z;
                    tmpArmor.angular_velocity_x = tmpArmor.internal_velocity_x * xCoefficient;
                    tmpArmor.angular_velocity_y = tmpArmor.internal_velocity_y * yCoefficient;
                } else {
                    tmpArmor.angular_velocity_x = 0;
                    tmpArmor.angular_velocity_y = 0;

                    tmpArmor.internal_velocity_y = 0;
                    tmpArmor.internal_velocity_x = 0;
                    tmpArmor.velocity_z = 0;
                }
                armors.push_back(tmpArmor);
            }
        }
    }
    return !armors.empty();
}

bool detect(const Mat &pSrcImage, const Rect &curSearchArea, const Armor &referenceArmor, Armor &armor,
            Ptr<cuda::LookUpTable> lookupTable, const float xCoefficient, const float yCoefficient,
            const float zCoefficient) {
    vector<Armor> armors;
    bool found = detect(pSrcImage, curSearchArea, &referenceArmor, armors, lookupTable, xCoefficient, yCoefficient,
                        zCoefficient);
    if (!found) return false;
    auto armorsCount = (unsigned int) armors.size();
    if (armorsCount > 1) {
        float minDifference =
                abs(armors[0].height - referenceArmor.height) + abs(armors[0].width - referenceArmor.width) +
                abs(armors[0].x - referenceArmor.x) + abs(armors[0].y - referenceArmor.y) +
                abs(armors[0].z - referenceArmor.z);
        int index = 0;
        for (unsigned int i = 1; i < armorsCount; i++) {
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
    } else {
        armor = armors[0];
    }
    return true;
}

void getSearchArea(vector<Armor> &armors, vector<Rect> &searchAreas, int width, int height) {
    auto size = (int) armors.size();
    for (int i = 0; i < size; i++) {
        Armor curArmor = armors[i];
        float x = curArmor.x - curArmor.width + curArmor.internal_velocity_x;
        float y = curArmor.y - curArmor.height + curArmor.internal_velocity_y;
        float saWidth = curArmor.width * 2;
        float saHeight = curArmor.height * 2;
        if (x + saWidth <= 0 || y + saHeight <= 0 || x >= width || y >= height) {
            searchAreas.erase(searchAreas.begin() + i);
            armors.erase(armors.begin() + i);
            size--;
            i--;
            continue;
        }
        if (x < 0) {
            saWidth = saWidth + x;
            x = 0;
        }
        if (x + saWidth > width) {
            saWidth = width - x;
        }
        if (y < 0) {
            saHeight = saHeight + y;
            y = 0;
        }
        if (y + saHeight > height) {
            saHeight = height - y;
        }
        if (saWidth == -7) {
            cout << "";
        }
        searchAreas[i] = Rect((int) round(x), (int) round(y), (int) round(saWidth), (int) round(saHeight));
    }
}

int main(int argc, char **argv) {

    Mat pSrcImage;
    FileStorage fs;
    if (argc > 1) {
        fs.open(argv[1], FileStorage::READ);
    } else {
        fs.open("config.xml", FileStorage::READ);
    }
    Settings settings;
    settings.read(fs);
    VideoCapture cap;

    if (settings.cameraID != -1) {
        cap.open(settings.cameraID);
    } else {
        cap.open(settings.fileName);
    }

    /// Camera setup
    cap.set(CV_CAP_PROP_FRAME_WIDTH, settings.width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, settings.height);
    cap.set(CV_CAP_PROP_FPS, settings.fps);
    cap.set(CV_CAP_PROP_BRIGHTNESS, settings.brightness);
    cap.set(CV_CAP_PROP_CONTRAST, settings.contrast);
    cap.set(CV_CAP_PROP_SATURATION, settings.saturation);
    cap.set(CV_CAP_PROP_HUE, settings.hue);
    cap.set(CV_CAP_PROP_GAIN, settings.hue);
    cap.set(CV_CAP_PROP_EXPOSURE, settings.exposure);

    cap >> pSrcImage;

#ifndef NDEBUG

    Mat pResultImage;

    if (nTargetColor == TARGET_BLUE) {
        sTargetColor = Scalar(255, 0, 0);
    } else {
        sTargetColor = Scalar(0, 0, 255);
    }

    namedWindow("Result image", WINDOW_AUTOSIZE);
//    setMouseCallback("Result image", CallBackFunc, nullptr);

    printf("width = %.2f\n", cap.get(CV_CAP_PROP_FRAME_WIDTH));
    printf("height = %.2f\n", cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    printf("fbs = %.2f\n", cap.get(CV_CAP_PROP_FPS));
    printf("brightness = %.2f\n", cap.get(CV_CAP_PROP_BRIGHTNESS));
    printf("contrast = %.2f\n", cap.get(CV_CAP_PROP_CONTRAST));
    printf("saturation = %.2f\n", cap.get(CV_CAP_PROP_SATURATION));
    printf("hue = %.2f\n", cap.get(CV_CAP_PROP_HUE));
    printf("exposure = %.2f\n", cap.get(CV_CAP_PROP_EXPOSURE));

    bool playVideo = true;
    clock_t totalTime = clock();
    long int frameCount = 0;
#endif //ifndef NDEBUG

    vector<Armor> armors;
    vector<Rect> searchAreas;
//    Size pSize = pSrcImage.size();
//    int type = pSrcImage.type();

    Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();

    int width = pSrcImage.cols;
    int height = pSrcImage.rows;

    float xCoefficient = settings.viewingAngleX / width * settings.fps;
    float yCoefficient = settings.viewingAngleY / height * settings.fps;

    int midX = width / 2;
    int midY = height / 2;

    Rect frame = Rect(0, 0, width, height);

#ifdef NDEBUG
    I2C<Armor> i2c("/dev/i2c-1", 0x04);
#endif

    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, 10) * 255.0);

    Ptr<cuda::LookUpTable> lut = cuda::createLookUpTable(lookUpTable);

    for (int tenFrame = 0; pSrcImage.data; tenFrame++) {
#ifndef NDEBUG
        frameCount++;
#endif

        auto searchAreaCount = (unsigned int) searchAreas.size();
        if (searchAreaCount == 0) {
            detect(pSrcImage, frame, nullptr, armors, lut, xCoefficient, yCoefficient, settings.zCoefficient);
            tenFrame = 0;
        } else {
            // TODO - Multi-threading
            for (unsigned int i = 0; i < searchAreaCount; i++) {
                Mat subImage(pSrcImage, searchAreas[i]);
                bool found = detect(subImage, searchAreas[i], armors[i], armors[i], lut, xCoefficient, yCoefficient,
                                    settings.zCoefficient);
                if (!found) {
                    armors.erase(armors.begin() + i);
                    searchAreas.erase(searchAreas.begin() + i);
                    searchAreaCount--;
                    i--;
                }
            }
            if (tenFrame == 11) {
                ///cover all the known armors with black rect
                for (unsigned int i = 0; i < searchAreaCount; i++) {
                    rectangle(pSrcImage, Rect((int) round(armors[i].x - armors[i].width / 2),
                                              (int) round(armors[i].y - armors[i].height / 2),
                                              (int) round(armors[i].width), (int) round(armors[i].height)),
                              Scalar(0, 0, 0), CV_FILLED);
                }
                vector<Armor> newArmors;
                bool found = detect(pSrcImage, frame, nullptr, newArmors, lut, xCoefficient, yCoefficient,
                                    settings.zCoefficient);
                if (found)
                    armors.insert(armors.end(), newArmors.begin(), newArmors.end());
                tenFrame = 0;
            }
        }

        auto numArmors = (unsigned int) armors.size();
        Armor *resultArmor = &armors[0];
        if (numArmors > 1) {
            float minScore = (float) (abs(armors[0].x - midX) + abs(armors[0].y - midY) + armors[0].z +
                                      pow(armors[0].internal_velocity_x, 2) + pow(armors[0].internal_velocity_y, 2) +
                                      pow(armors[0].velocity_z, 2));
            for (unsigned int i = 1; i < numArmors; i++) {
                //TODO - score formula
                float score = (float) (abs(armors[i].x - midX) + abs(armors[i].y - midY) + armors[i].z +
                                       pow(armors[i].internal_velocity_x, 2) + pow(armors[i].internal_velocity_y, 2) +
                                       pow(armors[i].velocity_z, 2)); // the smaller the better
                if (score < minScore) {
                    minScore = score;
                    resultArmor = &armors[i];
                }
            }
        }
#ifdef NDEBUG
        i2c.send(*resultArmor);
#endif
        searchAreas.resize(numArmors);
        getSearchArea(armors, searchAreas, width, height);


#ifndef NDEBUG
        if (playVideo) {
            pSrcImage.copyTo(pResultImage);
            for (unsigned int i = 0; i < numArmors; i++) {
                circle(pResultImage, Point(armors[i].x, armors[i].y), 5, sTargetColor, CV_FILLED);
                rectangle(pResultImage, searchAreas[i], sTargetColor, 1);
                putText(pResultImage, to_string(armors[i].z) + " m", Point(armors[i].x, armors[i].y),
                        FONT_HERSHEY_SIMPLEX,
                        1, sTargetColor, 2);
                putText(pResultImage, "vx: " + to_string(armors[i].angular_velocity_x) + " rad/s",
                        Point(armors[i].x, armors[i].y + 25), FONT_HERSHEY_SIMPLEX,
                        1, sTargetColor, 2);
                putText(pResultImage, "vy: " + to_string(armors[i].angular_velocity_y) + " rad/s",
                        Point(armors[i].x, armors[i].y + 50), FONT_HERSHEY_SIMPLEX,
                        1, sTargetColor, 2);
                putText(pResultImage, "vz: " + to_string(armors[i].velocity_z) + " m/s",
                        Point(armors[i].x, armors[i].y + 75),
                        FONT_HERSHEY_SIMPLEX,
                        1, sTargetColor, 2);
            }
        }
        imshow("Result image", pResultImage);
#ifdef FBF
        char c = (char) waitKey(0);
#else
        char c = (char) waitKey(1);
#endif
        if (c == 27)
            break;
        else if (c == ' ')
            playVideo = !playVideo;
//        else if(!numArmors)
//            cin.get();
        if (playVideo)
#endif //ifndef NDEBUG
            cap >> pSrcImage;
    }
#ifndef NDEBUG
    cout << "average time: " << (double) (clock() - totalTime) / CLOCKS_PER_SEC / frameCount << "s, FPS: "
         << frameCount / (double) (clock() - totalTime) * CLOCKS_PER_SEC << endl;
#endif
    cap.release();
    return 0;
}
