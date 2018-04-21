#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

class Settings {

public:
    string input;
    string fileName;
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

    void write(FileStorage &fs) const {
        fs << "input" << input
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

    void read(const FileStorage &fs) {
        fs["input"] >> input;
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
};

int main() {
    Settings settings;
    settings.input = "1";
    settings.width = 640;
    settings.height = 480;
    settings.fps = 30;
    settings.viewingAngleY = 1.81716;
    settings.viewingAngleX = 1.81716;
    settings.zCoefficient = 275;
    settings.brightness = 0;
    settings.contrast = 0.5;
    settings.saturation = 0.5;
    settings.hue = 0.5;
    settings.gain = 0;
    settings.exposure = 0.05;
    settings.isFishEye = false;
//
//    FileStorage fs("config.xml", FileStorage::WRITE);
//    settings.write(fs);

    Settings setting;
    FileStorage fsd("config.xml", FileStorage::READ);
    setting.read(fsd);

    cout << setting.input << endl;
    return 0;
}