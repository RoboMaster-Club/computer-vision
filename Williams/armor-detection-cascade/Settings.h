//
// Created by why on 5/8/18.
//

#ifndef ARMOR_DETECTION_SETTINGS_H
#define ARMOR_DETECTION_SETTINGS_H

#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Settings {

public:
    string input;
    string fileName;
    string cascade;
    int cameraID;
    int width;
    int height;
    int fps;
    float viewingAngleX;
    float viewingAngleY;
    float zCoefficient;
    int brightness;
    float contrast;
    float saturation;
    float hue;
    float gain;
    float exposure;
    bool isFishEye;

    void write(FileStorage &fs) const;

    void read(const FileStorage &fs);
};


#endif //ARMOR_DETECTION_SETTINGS_H
