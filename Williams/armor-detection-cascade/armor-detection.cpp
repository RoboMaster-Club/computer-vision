#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

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

void CallBackFunc(int event, int x, int y, int flags, void *userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point p = Point(x, y);
        cout << ((Mat *) userdata)->at<Vec3b>(p) << endl;
    }
}

#endif

bool detectMultiple(const Mat &pSrcImage, const Rect &curSearchArea, const Armor *referenceArmor, vector<Armor> &armors,
                    Ptr<cuda::CascadeClassifier> cascade, const float xCoefficient, const float yCoefficient,
                    const float zCoefficient) {
    cuda::GpuMat gSrcImage, gGray, objBuf;
    gSrcImage.upload(pSrcImage);
    cuda::cvtColor(gSrcImage, gGray, COLOR_BGR2GRAY);
    cuda::equalizeHist(gGray, gGray);

//    Mat tmp;
//    gGray.download(tmp);
//    imshow("tmp", tmp);
//
//    waitKey(0);

    vector<Rect> armorRect;

    cascade->detectMultiScale(gGray, objBuf);
    cascade->convert(objBuf, armorRect);

    auto armorSize = (unsigned int) armorRect.size();

    Armor tmpArmor;
    for (unsigned i = 0; i < armorSize; i++) {
        tmpArmor.width = armorRect[i].width;
        tmpArmor.height = armorRect[i].height;
        tmpArmor.x = curSearchArea.tl().x + armorRect[i].x + (float) armorRect[i].width / 2;
        tmpArmor.y = curSearchArea.tl().y + armorRect[i].y + (float) armorRect[i].height / 2;
        tmpArmor.z = zCoefficient / armorRect[i].height;
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
    return !armors.empty();
}

bool detectSingle(const Mat &pSrcImage, const Rect &curSearchArea, const Armor &referenceArmor, Armor &armor,
                  Ptr<cuda::CascadeClassifier> cascade, const float xCoefficient, const float yCoefficient,
                  const float zCoefficient) {
    vector<Armor> armors;
    if (!detectMultiple(pSrcImage, curSearchArea, &referenceArmor, armors, cascade, xCoefficient, yCoefficient,
                        zCoefficient))
        return false;
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
        if (y + saHeight > height - 1) {
            saHeight = height - y;
        }
        if (saWidth == -7) {
            cout << "";
        }
        searchAreas[i] = Rect((int) round(x), (int) round(y), (int) round(saWidth), (int) round(saHeight));
    }
}

void outputResult(Armor *resultArmor) {
//    i2c.send(*resultArmor);
    printf("x: %f, y: %f, z: %fm, vx: %frad/s, vy: %frad/s, vz: %fm/s\n", resultArmor->x, resultArmor->y, resultArmor->z, resultArmor->angular_velocity_x, resultArmor->angular_velocity_y, resultArmor->velocity_z);
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

    int width = pSrcImage.cols - 1;
    int height = pSrcImage.rows - 1;

    float xCoefficient = settings.viewingAngleX / width * settings.fps;
    float yCoefficient = settings.viewingAngleY / height * settings.fps;

    int midX = width / 2;
    int midY = height / 2;
    long foundCount = 0;

    Rect frame = Rect(0, 0, width, height);

#ifdef NDEBUG
    I2C<Armor> i2c("/dev/i2c-1", 0x04);
#endif

    Ptr<cuda::CascadeClassifier> cascade = cuda::CascadeClassifier::create(settings.cascade);
    //cascade->setFindLargestObject(true);
    //cascade->setScaleFactor(1.01);
    //cascade->setMinNeighbors(0);
//    cascade->setScaleFactor(1.01);
    for (int tenFrame = 0; pSrcImage.data; tenFrame++) {
#ifndef NDEBUG
        frameCount++;
#endif

        auto searchAreaCount = (unsigned int) searchAreas.size();
        if (searchAreaCount == 0) {
            detectMultiple(pSrcImage, frame, nullptr, armors, cascade, xCoefficient, yCoefficient,
                           settings.zCoefficient);
            tenFrame = 0;
        } else {
            // TODO - Multi-threading
            for (unsigned int i = 0; i < searchAreaCount; i++) {
                Mat subImage(pSrcImage, searchAreas[i]);
                bool found = detectSingle(subImage, searchAreas[i], armors[i], armors[i], cascade, xCoefficient,
                                          yCoefficient, settings.zCoefficient);
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
                bool found = detectMultiple(pSrcImage, frame, nullptr, newArmors, cascade, xCoefficient, yCoefficient,
                                            settings.zCoefficient);
                if (found)
                    armors.insert(armors.end(), newArmors.begin(), newArmors.end());
                tenFrame = 0;
            }
        }

        auto numArmors = (unsigned int) armors.size();
        Armor *resultArmor = &armors[0];
	if (numArmors > 0) {
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
	    outputResult(resultArmor);
	    foundCount++;
	}


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
    cout << "detection rate: " << (double)foundCount / frameCount << endl;
#endif
    cap.release();
    return 0;
}
