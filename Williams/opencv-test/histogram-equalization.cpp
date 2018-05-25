#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

void CallBackFunc(int event, int x, int y, int flags, void *userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point p = Point(x, y);
        cout << ((Mat*)userdata)->at<Vec3b>(p) << endl;
    }
}

int main(int argc, char *argv[]) {
    VideoCapture cap;
    if (argc == 2) {
        cap.open(argv[1]);
    } else {
        cap.open("/home/why/robot_red_5m_480p.mp4");
    }
    if (!cap.isOpened()) {
        printf("No image data \n");
        return -1;
    }

    Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, 10) * 255.0);

    namedWindow("original image", WINDOW_AUTOSIZE);
    namedWindow("Gamma correction", WINDOW_AUTOSIZE);
    Mat src, dst, pHSV;
    setMouseCallback("original image", CallBackFunc, &pHSV);
    setMouseCallback("Gamma correction", CallBackFunc, &pHSV);
    cap >> src;
    while(src.data) {
        imshow("original image", src);

        src.copyTo(dst);

//        /// histogram equalization
//        cvtColor(dst, dst, COLOR_BGR2YCrCb);
//        vector<Mat> channels;
//        split(dst, channels);
//        equalizeHist(channels[0], channels[0]);
//        merge(channels, dst);
//        cvtColor(dst, dst, COLOR_YCrCb2BGR);
//        imshow("Result", dst);

        /// gamma correction
        LUT(dst, lookUpTable, dst);
        imshow("Gamma correction", dst);

        cvtColor(dst, pHSV, COLOR_BGR2HSV);

        waitKey(0);
        cap >> src;
    }
}