#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

cv::Mat hsv_image;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        Point p = Point(x,y);
        cout << hsv_image.at<Vec3b>(p) << endl;
    }
}

int main() {
    cv::namedWindow("test");
    setMouseCallback("test", CallBackFunc, NULL);
    while(true) {
        cv::Mat bgr_image = cv::imread("/home/why/test1.png");
        // Convert input image to HSV

        cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
        // Threshold the HSV image, keep only the red pixels
        cv::Mat lower_red_hue_range;
        cv::Mat upper_red_hue_range;
        cv::inRange(hsv_image, cv::Scalar(0, 0, 200), cv::Scalar(40, 100, 256), lower_red_hue_range);
        cv::inRange(hsv_image, cv::Scalar(200, 0, 200), cv::Scalar(256, 50, 256), upper_red_hue_range);
        cv::imshow("test", lower_red_hue_range | upper_red_hue_range);
        char c = (char) waitKey(1);
        if (c == 27)
            break;
    }
}

