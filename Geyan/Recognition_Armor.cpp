#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>
using namespace cv;
using namespace std;
#define THREADS 4
#define ANGLE_THRE 10
#define H_RATIO 10
#define PI 3.14159265358979
#define PATH "E:\\Purdue\\VS\\C++\\Dji-RM\\testVideo\\5-1.mp4"
#define bRED true
#if bRED
#define COLOR Scalar(0,0,255)
#else
#define COLOR Scalar(255,0,0)
#endif
Mat clrRng(1, 256, CV_8U);
float tmpD = -1;
typedef struct Armor {
	int id;
	float w, h;
	double dx, dy, d, v;
} Armor;
Mat Proc(Mat srcImg, Mat &dstImg, Mat & HSV) {
	Mat Ctr, Brt, Clr, lClr, rClr;
	uchar *c = clrRng.ptr();
	for (int i = 0; i < 256; c[i] = saturate_cast<uchar>(pow(i++ / 255.0, 4) * 255.0));
	LUT(srcImg, clrRng, dstImg);
	Clr = Brt = Mat::zeros(srcImg.size(), CV_8UC1);
	cvtColor(dstImg, HSV, COLOR_BGR2HSV);
	inRange(HSV, Scalar(0, 0, 200), Scalar(200, 200, 255), Brt);
	if (bRED) {
		inRange(HSV, Scalar(0, 100, 100), Scalar(10, 255, 255), lClr);
		inRange(HSV, Scalar(170, 100, 100), Scalar(180, 255, 255), rClr);
		Clr = lClr | rClr;
	}
	else inRange(HSV, Scalar(120, 100, 100), Scalar(140, 255, 255), Clr);
	blur((Brt |= Clr), Ctr, Size(3, 3));
	Canny(Ctr, Ctr, 100, 200);
	return Ctr;
}
void Draw(Mat Img, Point *pt, Armor armor) {
	line(Img, pt[0], pt[2], COLOR, 2, 8, 0);
	line(Img, pt[2], pt[3], COLOR, 2, 8, 0);
	line(Img, pt[3], pt[1], COLOR, 2, 8, 0);
	line(Img, pt[1], pt[0], COLOR, 2, 8, 0);
	putText(Img, "v: " + to_string(armor.v), Point(armor.dx, armor.dy + 40), FONT_ITALIC, 1, COLOR, 2);
	putText(Img, "d: " + to_string(armor.d), Point(armor.dx, armor.dy - 40), FONT_ITALIC, 1, COLOR, 2);
}
void ArmorDetect(Mat & Img, vector<RotatedRect> ellipses, vector<Armor> & armors) {
	RotatedRect pre, pos;
	omp_set_num_threads(THREADS);
#pragma omp parallel for
	for (int i = 0; i < ellipses.size() - 1; ++i)
		for (int j = i + 1; j < ellipses.size(); ++j) {
			pre = ellipses[i];
			pos = ellipses[j];
			if ((abs(pre.angle - pos.angle) < ANGLE_THRE || 180 - abs(pre.angle - pos.angle) < ANGLE_THRE) && abs(pre.size.height - pos.size.height) / (pre.size.height + pos.size.height) < 0.05 && abs(pre.size.width - pos.size.width) / (pre.size.width + pos.size.width) < 0.05 && abs(pre.center.x - pos.center.x) / (pre.size.height + pos.size.height) > 0.5 && abs(pre.center.x - pos.center.x) / (pre.size.height + pos.size.height) < 3 && abs(pre.center.y - pos.center.y) / (pre.size.height + pos.size.height) < 0.3) {
				Armor armor;
				if (armors.size() > 0) armor.id = armors[armors.size() - 1].id + 1;
				else armor.id = -1;
				if (pre.center.x > pos.center.x) swap(pre, pos);
				armor.w = pos.center.x - pre.center.x;
				armor.h = pre.size.height + pos.size.height;
				armor.d = 300 / (pre.size.height + pos.size.height);
				if (tmpD < 0) tmpD = armor.d;
				Point pt[4];
				pt[0] = Point(pre.center.x - sin(pre.angle * PI / 180) * pre.size.height, pre.center.y + cos(pre.angle * PI / 180) * pre.size.height);
				pt[1] = Point(pre.center.x + sin(pre.angle * PI / 180) * pre.size.height, pre.center.y - cos(pre.angle * PI / 180) * pre.size.height);
				pt[2] = Point(pos.center.x - sin(pos.angle * PI / 180) * pos.size.height, pos.center.y + cos(pos.angle * PI / 180) * pos.size.height);
				pt[3] = Point(pos.center.x + sin(pos.angle * PI / 180) * pos.size.height, pos.center.y - cos(pos.angle * PI / 180) * pos.size.height);
				if (pt[0].y > pt[1].y) swap(pt[0], pt[1]);
				if (pt[2].y > pt[3].y) swap(pt[2], pt[3]);
				for (int k = 0; k < 4; ++k) {
					armor.dx += pt[i].x / 4;
					armor.dy += pt[i].y / 4;
				}
				if (armor.dx > 4000 || armor.dx < 1 || armor.dy>2000 || armor.dy < 1) continue;
				//if(armors.size()>1)
				armor.v = armor.d - tmpD;
				printf("X: %f; Y: %f; D: %f; V: %f, preD: %f, curD: %f\n", armor.dx, armor.dy, armor.d, armor.v, tmpD, armor.d);
				armors.push_back(armor);
				tmpD = armor.d;
				Draw(Img, pt, armor);
			}
		}
}
int main(int argc, char **argv) {
	Mat srcImg, dstImg, HSV;
	VideoCapture cap = VideoCapture(PATH);
	//cap.open(2);
	cap >> srcImg;
	while (srcImg.data) {
		vector<Vec4i> tmp4I;
		vector<vector<Point>> tmpCtr;
		findContours(Proc(srcImg, dstImg, HSV), tmpCtr, tmp4I, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<RotatedRect> Elp;
		omp_set_num_threads(THREADS);
#pragma omp parallel for
		for (int i = 0; i < tmpCtr.size(); ++i) {
			int red = 0, blue = 0;
			if (tmpCtr[i].size() >= 5) {
				for (int j = 0; j < tmpCtr[i].size(); ++j)
					if (HSV.at<Vec3b>(tmpCtr[i][j])[0] <= 60) ++red;
					else if (HSV.at<Vec3b>(tmpCtr[i][j])[0] >= 180 && HSV.at<Vec3b>(tmpCtr[i][j])[0] <= 240) ++blue;
					if ((bRED && red <= blue) || (!bRED && blue <= red)) continue;
					RotatedRect tmp = fitEllipse((tmpCtr[i]));
					float hw = tmp.size.height / tmp.size.width;
					if ((tmp.angle < ANGLE_THRE || 180 - tmp.angle < ANGLE_THRE) && hw > 2 && hw < 4) Elp.push_back(tmp);
			}
		}
		vector<Armor> armors;
		ArmorDetect(dstImg, Elp, armors);
		namedWindow("Result image", WINDOW_AUTOSIZE);
		imshow("Result image", dstImg);
		if (waitKey(20) == 10) break;
		cap >> srcImg;
	}
}
