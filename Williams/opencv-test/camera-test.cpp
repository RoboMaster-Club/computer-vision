#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    VideoCapture cap(1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CV_CAP_PROP_FPS, 30);
    cap.set(CV_CAP_PROP_BRIGHTNESS, 0);
    cap.set(CV_CAP_PROP_CONTRAST, 0.5);
    cap.set(CV_CAP_PROP_SATURATION, 0.5);
    cap.set(CV_CAP_PROP_HUE, 0.5);
    cap.set(CV_CAP_PROP_GAIN, 0);
    cap.set(CV_CAP_PROP_EXPOSURE, 0.25);

    printf("width = %.2f\n",cap.get(CV_CAP_PROP_FRAME_WIDTH));
    printf("height = %.2f\n",cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    printf("fbs = %.2f\n",cap.get(CV_CAP_PROP_FPS));
    printf("brightness = %.2f\n",cap.get(CV_CAP_PROP_BRIGHTNESS));
    printf("contrast = %.2f\n",cap.get(CV_CAP_PROP_CONTRAST));
    printf("saturation = %.2f\n",cap.get(CV_CAP_PROP_SATURATION));
    printf("hue = %.2f\n",cap.get(CV_CAP_PROP_HUE));
    printf("exposure = %.2f\n",cap.get(CV_CAP_PROP_EXPOSURE));

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