#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <omp.h>
#define T_ANGLE_THRE 10
#define T_SIZE_THRE 5
using namespace cv;
using namespace std;
void brightAdjust(IplImage* src, IplImage* dst, double contrast, double bright) {
	unsigned char* srcData = (unsigned char*)src->imageData, *dstData = (unsigned char*)dst->imageData;
	int tmpVal, step = src->widthStep / sizeof(unsigned char) / 3;
	omp_set_num_threads(8);
	for (int i = 0; i<src->height; ++i)
		for (int j = 0; j <src->width; ++j)
			for (int k = 0; k < 3; ++k) {
				tmpVal = (int)(contrast * srcData[(i*step + j) * 3 + k] + bright);
				tmpVal = (tmpVal < 0) ? 0 : (tmpVal > 255) ? 255 : tmpVal;
				dstData[(i*step + j) * 3 + k] = tmpVal;
			}
}
void getDiffImg(IplImage* src1, IplImage* src2, IplImage* dst, int Thre) {
	unsigned char *srcData1 = (unsigned char*)src1->imageData, *srcData2 = (unsigned char*)src2->imageData, *dstData = (unsigned char*)dst->imageData;
	int step = src1->widthStep / sizeof(unsigned char);
	omp_set_num_threads(8);
	for (int i = 0; i < src1->height; ++i)
		for (int j = 0; j < src1->width; ++j)
			dstData[i*step + j] = (srcData1[i*step + j] - srcData2[i*step + j] > Thre) ? 255 : 0;
}
vector<CvBox2D> armorDetect(vector<CvBox2D> ellipse) {
	vector<CvBox2D> rlt;
	CvBox2D armor;
	int m, n;
	double angle;
	rlt.clear();
	if (ellipse.size() < 2) return rlt;
	for (unsigned int i = 0; i < ellipse.size() - 1; ++i)
		for (unsigned int j = i + 1; j < ellipse.size(); ++j) {
			angle = abs(ellipse[i].angle - ellipse[j].angle);
			for (; angle > 180; angle -= 180);
			if ((angle < T_ANGLE_THRE || 180 - angle < T_ANGLE_THRE) && abs(ellipse[i].size.height - ellipse[j].size.height) < (ellipse[i].size.height + ellipse[j].size.height) / T_SIZE_THRE && abs(ellipse[i].size.width - ellipse[j].size.width) < (ellipse[i].size.width + ellipse[j].size.width) / T_SIZE_THRE) {
				armor.center.x = (ellipse[i].center.x + ellipse[j].center.x) / 2;
				armor.center.y = (ellipse[i].center.y + ellipse[j].center.y) / 2;
				armor.angle = (ellipse[i].angle + ellipse[j].angle) / 2;
				armor.angle += (180 - angle < T_ANGLE_THRE) ? 90 : 0;
				m = (ellipse[i].size.height + ellipse[j].size.height) / 2;
				n = sqrt((ellipse[i].center.x - ellipse[j].center.x) * (ellipse[i].center.x - ellipse[j].center.x) + (ellipse[i].center.y - ellipse[j].center.y) * (ellipse[i].center.y - ellipse[j].center.y));
				if (m < n) {
					armor.size.height = m;
					armor.size.width = n;
				}
				else {
					armor.size.height = n;
					armor.size.width = m;
				}
				rlt.push_back(armor);
			}
		}
	return rlt;
}
void draw(CvBox2D box, IplImage* img) {
	CvPoint2D32f point[4];
	int i;
	for (i = 0; i<4; ++i) {
		point[i].x = 0;
		point[i].y = 0;
	}
	cvBoxPoints(box, point);
	CvPoint pt[4];
	for (i = 0; i<4; ++i) {
		pt[i].x = (int)point[i].x;
		pt[i].y = (int)point[i].y;
	}
	cvLine(img, pt[0], pt[1], CV_RGB(0, 255, 0), 2, 8, 0);
	cvLine(img, pt[1], pt[2], CV_RGB(0, 255, 0), 2, 8, 0);
	cvLine(img, pt[2], pt[3], CV_RGB(0, 255, 0), 2, 8, 0);
	cvLine(img, pt[3], pt[0], CV_RGB(0, 255, 0), 2, 8, 0);
}
int main() {
	CvCapture *cap = cvCreateFileCapture("E:\\Purdue\\Fuck 3.8\\Robo\\RedCar.avi");
	IplImage *frame = NULL;
	CvSize imgSize;
	CvBox2D s;
	vector<CvBox2D> ellipse, rlt, armor;
	CvScalar sl;
	bool flag = false;
	CvSeq *contour = NULL, *lines = NULL;
	CvMemStorage *storage = cvCreateMemStorage(0);
	frame = cvQueryFrame(cap);
	imgSize = cvGetSize(frame);
	IplImage *rawImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 3), *grayImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1), *rImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1), *gImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1), *bImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1), *binImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1), *rltImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
	while (frame) {
		brightAdjust(frame, rawImg, 1, -120);
		cvSplit(rawImg, bImg, gImg, rImg, 0);
		getDiffImg(rImg, gImg, binImg, 25);
		cvDilate(binImg, grayImg, NULL, 3);
		cvErode(grayImg, rltImg, NULL, 1);
		cvFindContours(rltImg, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		for (; contour != NULL; contour = contour->h_next)
			if (contour->total > 10) {
				flag = true;
				s = cvFitEllipse2(contour);
				for (int i = 0; i < 5; ++i)
					for (int j = 0; j < 5; ++j)
						if (s.center.y - 2 + j > 0 && s.center.y - 2 + j < 480 && s.center.x - 2 + i > 0 && s.center.x - 2 + i <  640) {
							sl = cvGet2D(frame, (int)(s.center.y - 2 + j), (int)(s.center.x - 2 + i));
							flag = (sl.val[0] < 200 || sl.val[1] < 200 || sl.val[2] < 200) ? false : true;
						}
				if (flag) ellipse.push_back(s);
			}
		rlt = armorDetect(ellipse);
		for (unsigned int i = 0; i < rlt.size(); draw(rlt[i++], frame));
		cvShowImage("armor", frame);
		if (cvWaitKey(20) == 10) break;
		frame = cvQueryFrame(cap);
		armor.clear();
		ellipse.clear();
		rlt.clear();
	}
	cvReleaseCapture(&cap);
	return 0;
}
