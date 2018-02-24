#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    VideoCapture cap(0);
    Mat input;
    bool pause = false;
    namedWindow("Camera");
    cap >> input;
    while (input.data) {
        if (!pause) {
            imshow("Camera", input);
            cap >> input;
        }
        char c = (char) waitKey(1);
        if (c == 27) break;
        else if (c == ' ') pause = !pause;
    }
    return 0;
}