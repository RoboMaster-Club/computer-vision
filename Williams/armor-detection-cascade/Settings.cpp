#include "Settings.h"

void Settings::write(FileStorage &fs) const {
    fs << "input" << input
       << "cascade" << cascade
       << "width" << width
       << "height" << height
       << "fps" << fps
       << "viewing_angle_x" << viewingAngleX
       << "viewing_angle_y" << viewingAngleY
       << "z_coefficient" << zCoefficient
       << "brightness" << brightness
       << "contrast" << contrast
       << "saturation" << saturation
       << "hue" << hue
       << "gain" << gain
       << "exposure" << exposure
       << "is_fisheye" << isFishEye;
}

void Settings::read(const FileStorage &fs) {
    fs["input"] >> input;
    fs["cascade"] >> cascade;
    fs["width"] >> width;
    fs["height"] >> height;
    fs["fps"] >> fps;
    fs["viewing_angle_x"] >> viewingAngleX;
    fs["viewing_angle_y"] >> viewingAngleY;
    fs["z_coefficient"] >> zCoefficient;
    fs["brightness"] >> brightness;
    fs["contrast"] >> contrast;
    fs["saturation"] >> saturation;
    fs["hue"] >> hue;
    fs["gain"] >> gain;
    fs["exposure"] >> exposure;
    fs["is_fisheye"] >> isFishEye;

    if (input[0] >= '0' && input[0] <= '9' && input.size() == 1) {
        cameraID = input[0] - '0';
    } else {
        fileName = input;
        cameraID = -1;
    }
}