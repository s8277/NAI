#include "stdafx.h"

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/video/background_segm.hpp"
#include <sstream>


using namespace cv;
using namespace std;

// global constants
const int V_WIDTH = 640;
const int V_HEIGHT = 480;
int H_MIN = 0;
int H_MAX = 255;// 179;
int S_MIN = 0;
int S_MAX = 255;
int V_MIN = 0;
int V_MAX = 255;
// global variables
String optionsWindow = "optionsWindow";
String previewWindow = "previewWindow";

CascadeClassifier haar_cascade;
// current analyzed frame 
Mat currentFrame;
// current filter
int selectedOption = 0;
// global settings for parameters
int thresholdValue = 100;
int thresholdType = 3;


// function declarations
void optionChanged(int status, void* data) {};
void applyOption();
void faceDetect();
void on_trackbar(int, void*) {}
void drawHandPlace();
void objectDetectMog2();
void skinMask();
void colorTest();
void newHandTracking();


const int MAX_NUM_OBJECTS = 30;
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = 200 * 200;
bool isObjectDetected = false;


void detectContours();
void detectObject();
void substractMove() {};
void colorDetect() {};
void showMenu() {};
void handTracking();
void detectFingest(vector<Point> contours, Point2f c, float r, Mat &threshold);
Mat applyMask();

void applyOption() {

	switch (selectedOption) {
	case 1: // object contours
		detectContours();
		break;
	case 2: // detect object
		detectObject();
		break;
	case 3: // movement substraction
		substractMove();
		break;
	case 4: // face detection
		faceDetect();
		break;
	case 5: // shirt color detect
		colorDetect();
		break;
	default:
		//showMenu();
		//detectObject();

		handTracking();
		break;
	}

}











///////////////////////////////////////

std::vector< Vec3b > pixelsH;
std::vector< Vec3b > pixelsS;
std::vector< Vec3b > pixelsV;

void colorTest() {
	Mat thresh, gray, hsv, hsvf, gr555, hls, hlsf;

	cvtColor(currentFrame, hsv, COLOR_BGR2HSV);

	inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hsv);

	imshow("gray", gray);

	/*	cvtColor(currentFrame, gr555, COLOR_HLS2BGR_FULL);
	cvtColor(gr555, gray, COLOR_BGR2GRAY);
	//	cvtColor(currentFrame, gr555, COLOR_BGR2BGR555);
	cvtColor(gr555, hsv, COLOR_BGR2HSV);
	cvtColor(gr555, hsvf, COLOR_BGR2HSV_FULL);
	cvtColor(gr555, hls, COLOR_BGR2HLS);
	cvtColor(gr555, hlsf, COLOR_BGR2HLS_FULL);

	Scalar lowerSkin = Scalar(0, 36, 83);
	Scalar upperSkin = Scalar(179, 151, 228);

	inRange(gray, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), gray);
	inRange(gr555, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), gr555);
	inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hsv);
	inRange(hsvf, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hsvf);
	inRange(hls, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hls);
	inRange(hlsf, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hlsf);


	imshow("gray", gray);
	imshow("gr555", gr555);
	imshow("hsv", hsv);
	imshow("hsvf", hsvf);
	imshow("hls", hls);
	imshow("hlsf", hlsf);*/
}

void newHandTracking() {
	Mat hsv, erodeElement, dilateElement;

	cvtColor(currentFrame, hsv, COLOR_BGR2HSV);

	//light
	Scalar lowerSkin = Scalar(0, 36, 83);
	Scalar upperSkin = Scalar(179, 151, 228);
	// dark
	//Scalar lowerSkin = Scalar(82, 0, 92);
	//Scalar upperSkin = Scalar(128, 121, 255);

	//	inRange(hsv, lowerSkin, upperSkin, hsv);
	inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hsv);

	//	Mat str_el = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//	morphologyEx(hsv, hsv, cv::MORPH_OPEN, str_el);
	//	morphologyEx(hsv, hsv, cv::MORPH_CLOSE, str_el);


	vector<vector<Point> > contours;
	vector<Vec4i> heirarchy;
	vector<Point2i> center;
	vector<int> radius;
	int minTargetRadius = 50;

	erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilateElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	erode(hsv, hsv, erodeElement);
	dilate(hsv, hsv, dilateElement);
	erode(hsv, hsv, erodeElement);
	dilate(hsv, hsv, dilateElement);

	cv::findContours(hsv.clone(), contours, heirarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	size_t count = contours.size();
	//size_t count = center.size();
	cv::Scalar blue(255, 0, 0);
	cv::Scalar green(0, 255, 0);
	cv::Scalar red(0, 0, 255);
	Vec3b lastColor, currentColor;


	for (int i = 0; i<count; i++)

	{
		cv::Point2f c;
		float r;
		cv::minEnclosingCircle(contours[i], c, r);
		int objectsPerLine = 0;

		if (r >= minTargetRadius)
		{
			//		center.push_back(c);
			//		radius.push_back(r);
			//		circle(hsv, center[i], radius[i], red, 3);

			if (c.y - r > 0) {
				// warunki naiwne dla dloni
				// czy na wysokosci srodka okregu jest tylko jeden obiekt? jedna zmiana czarny-biały

				// srodek 0r
				for (int x = c.x - r; x < c.x + r; x++) {
					currentColor = hsv.at<cv::Vec3b>(c.y, x);
					if (lastColor[0] == 0.0 && currentColor[0] == 255.0) {
						objectsPerLine++;
					}
					lastColor = currentColor;
				}
				if (objectsPerLine > 2) {
					circle(currentFrame, c, r, green, 3);
					// to nie dlon
					continue;
				}
				objectsPerLine = 0;
				// nadgarstek -0.5r
				for (int x = c.x - r; x < c.x + r; x++) {
					currentColor = hsv.at<cv::Vec3b>(c.y - r / 2, x);
					if (lastColor[0] == 0.0 && currentColor[0] == 255.0) {
						objectsPerLine++;
					}
					lastColor = currentColor;
				}
				if (objectsPerLine > 1) {
					circle(currentFrame, c, r, blue, 3);
					// to nie dlon
					continue;
				}
				// palce 0.5r
				circle(currentFrame, c, r, red, 3);
				objectsPerLine = 0;
				for (int x = c.x - r; x < c.x + r; x++) {
					currentColor = hsv.at<cv::Vec3b>(c.y - r / 2, x);
					if (lastColor[0] == 0.0 && currentColor[0] == 255.0) {
						objectsPerLine++;
					}
					lastColor = currentColor;
				}
				std::cout << "Wykrylem " << objectsPerLine << " palcow" << endl;
				// czy na wysokosci ponizej srodka okregu jest tylko jeden obiekt? jedna zmiana czarny-biały, biały-czarny
				// czy na wysokosci powyzej srodka okregu jest zero lub wiecej obiektow?
				//	int y = c.y - r / 2;
				//	for (int x = c.x - r; x < c.x + r; x++) {
				//		std::cout << hsv.at<cv::Vec3b>(y, x) << ", ";
				//	}
			}
			std::cout << std::endl << std::endl;
		}

	}

	imshow("hsv", hsv);


}

void skinMask() {
	Mat thresh, erodeElement, dilateElement, mask, hsv;
	Scalar lowerSkin = Scalar(0, 20, 127);
	Scalar upperSkin = Scalar(23, 113, 255);

	//	lowerSkin = Scalar(0, 36, 83);
	//	upperSkin = Scalar(179, 151, 228);



	cvtColor(currentFrame, hsv, COLOR_BGR2HSV_FULL);

	inRange(hsv, lowerSkin, upperSkin, mask);

	erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
	erode(mask, mask, erodeElement);
	dilate(mask, mask, dilateElement);

	bitwise_and(currentFrame, currentFrame, thresh, mask = mask);

	imshow("hsv", hsv);
	imshow("maska", mask);
	imshow("result", thresh);
}

bool handFound = false;
Scalar skinMin = Scalar();
Scalar skinMax = Scalar();

void objectDetectMog2() {

	//Mat img = imread("../data/fruits.jpg");
	//	Mat mask = np.zeros(img.shape[:2], np.uint8);

	//	imshow("Coostam", mask);

	/*	VideoCapture input(0);
	//Mat frame; //current frame

	Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method

	Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor

	pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach

	pMOG2->apply(currentFrame, fgMaskMOG2);


	stringstream ss;
	rectangle(currentFrame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255), -1);
	ss << input.get(CAP_PROP_POS_FRAMES);
	string frameNumberString = ss.str();
	putText(currentFrame, frameNumberString.c_str(), cv::Point(15, 15),
	FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	//show the current frame and the fg masks
	imshow("Inne", currentFrame);*/
}

void drawHandPlace() {
	Mat frameHsv;
	Mat thresh, erodeElement, dilateElement, mask;
	cvtColor(currentFrame, frameHsv, CV_BGR2HSV);
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);
	Scalar lowerSkin = Scalar(H_MIN, S_MIN, V_MIN);
	Scalar upperSkin = Scalar(H_MAX, S_MAX, V_MAX);

	Scalar lowerWall = Scalar(18, 0, 113);
	Scalar upperWall = Scalar(100, 56, 193);

	putText(currentFrame, "Umiesc reke w wyznaczonym obszarze", Point(0, 20), 2, 1, Scalar(0, 255, 0), 2);
	ellipse(currentFrame, Point(450, 250), Size(60, 60), 0, 0, 360, green, 2, 8);
	std::vector<Vec3b> colorPoints;
	colorPoints.push_back(frameHsv.at<cv::Vec3b>(220, 420));
	colorPoints.push_back(frameHsv.at<cv::Vec3b>(220, 480));
	colorPoints.push_back(frameHsv.at<cv::Vec3b>(280, 420));
	colorPoints.push_back(frameHsv.at<cv::Vec3b>(280, 480));
	colorPoints.push_back(frameHsv.at<cv::Vec3b>(250, 450));

	ellipse(currentFrame, Point(420, 220), Size(2, 2), 0, 0, 360, green, 2, 8);
	ellipse(currentFrame, Point(480, 220), Size(2, 2), 0, 0, 360, green, 2, 8);
	ellipse(currentFrame, Point(420, 280), Size(2, 2), 0, 0, 360, green, 2, 8);
	ellipse(currentFrame, Point(480, 280), Size(2, 2), 0, 0, 360, green, 2, 8);
	ellipse(currentFrame, Point(450, 250), Size(2, 2), 0, 0, 360, green, 2, 8);

	for (int i = 0; i < colorPoints.size(); i++) {
		if (i == 0) {
			lowerSkin = colorPoints[0];
			upperSkin = colorPoints[0];
		}
		else {
			if (lowerSkin[2] > colorPoints[i][2]) {
				lowerSkin = colorPoints[i];
			}
			if (upperSkin[2] < colorPoints[i][2]) {
				upperSkin = colorPoints[i];
			}
		}
	}

	//lowerWall[1] = lowerSkin[1];
	//upperWall[1] = upperSkin[1];

	lowerSkin = lowerWall;
	upperSkin = upperWall;
	std::cout << lowerSkin << std::endl;
	std::cout << upperSkin << std::endl;

	/*	setTrackbarPos("H_MIN", optionsWindow, lowerSkin[0]);
	setTrackbarPos("S_MIN", optionsWindow, lowerSkin[1]);
	setTrackbarPos("V_MIN", optionsWindow, lowerSkin[2]);
	setTrackbarPos("H_MAX", optionsWindow, upperSkin[0]);
	setTrackbarPos("S_MAX", optionsWindow, upperSkin[1]);
	setTrackbarPos("V_MAX", optionsWindow, upperSkin[2]);
	*/
	//	lowerSkin = Scalar(0, 36, 83);
	//	upperSkin = Scalar(179, 151, 228);

	inRange(frameHsv, lowerSkin, upperSkin, mask);
	//inRange(frameHsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), mask);

	erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(mask, mask, erodeElement);
	dilate(mask, mask, dilateElement);


	bitwise_and(currentFrame, currentFrame, thresh, mask = mask);

	imshow("hsv", frameHsv);
	imshow("maska", mask);
	imshow("result", thresh);
}

void drawHandPlace2() {
	Mat frameHsv;
	cvtColor(currentFrame, frameHsv, CV_BGR2HSV);
	putText(currentFrame, "Umiesc reke w wyznaczonym obszarze", Point(0, 20), 2, 1, Scalar(0, 255, 0), 2);
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);
	std::vector <Point> handPoints;
	std::vector <Vec3b> colorPoints;
	handPoints.push_back(Point(450, 250));
	//	handPoints.push_back(Point(320, 200));
	//	handPoints.push_back(Point(390, 90));
	//	handPoints.push_back(Point(450, 60));
	//	handPoints.push_back(Point(500, 80));
	//	handPoints.push_back(Point(550, 110));
	bool allOk = true, colorMatch = false;

	//	cv::Vec3b pixel = frameHsv.at<cv::Vec3b>(handPoints[0].y, handPoints[0].x);
	//	putText(currentFrame, intToString(handPoints[0].x) + "x" + intToString(handPoints[0].y) + " - " + intToString(pixel[0]) + "," + intToString(pixel[1]) + "," + intToString(pixel[2]), Point(0, 50), 2, 1, pixel, 2);

	for (int i = 0; i < handPoints.size(); i++) {
		colorMatch = false;
		cv::Vec3b pixel = currentFrame.at<cv::Vec3b>(handPoints[i].y, handPoints[i].x);
		colorPoints.push_back(pixel);
		if (i > 0) {
			if (
				colorPoints[0][0] > pixel[0] - 20 && colorPoints[0][0] < pixel[0] + 20
				//	colorPoints[0][1] > pixel[1] - 20 && colorPoints[0][1] < pixel[1] + 20 &&
				//	colorPoints[0][2] > pixel[2] - 20 && colorPoints[0][2] < pixel[2] + 20 
				) {
				colorMatch = true;
			}
			else {
				allOk = false;
			}
		}
		ellipse(currentFrame, Point(handPoints[i].x, handPoints[i].y), i > 0 ? Size(20, 20) : Size(30, 30), 0, 0, 360, colorMatch ? green : red, 2, 8);

		putText(currentFrame, to_string(pixel[0]) + "," + to_string(pixel[1]) + "," + to_string(pixel[2]), Point(handPoints[i].x - 10, handPoints[i].y + 20), 2, 1, pixel, 1, 4);



		//	line(currentFrame, Point(handPoints[i].x, handPoints[i].y), Point(V_WIDTH / 2 + 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);
	}

	//	line(currentFrame, Point(V_WIDTH / 2 - 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 + 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);
	//	line(currentFrame, Point(V_WIDTH / 2 + 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 - 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);

}

void detectObject() {
	Mat frameHSV;
	Mat frameThreshold;

	threshold(currentFrame, currentFrame, 180, 255, thresholdType);

	cvtColor(currentFrame, frameHSV, COLOR_BGR2HSV);
	int minH = 179, maxH = 0, minS = 255, maxS = 0, minV = 255, maxV = 0;
	if (!isObjectDetected) {

		//		if (pixelsH.size() < 200) {

		/* area pipe
		std::vector<Point> contours;
		contours.push_back(Point(230, 230));
		contours.push_back(Point(230, 280));
		contours.push_back(Point(280, 230));
		contours.push_back(Point(280, 280));
		Mat pipeArea(currentFrame.clone());
		drawContours(pipeArea, contours, -1, Scalar(255, 255, 255));
		imshow("test",pipeArea);
		*/


		//		int minH = 179, maxH = 0, minS = 255, maxS = 0, minV = 255, maxV = 0;

		for (int x = V_WIDTH / 2 - 20; x < V_WIDTH / 2 + 20; x++) {
			for (int y = V_HEIGHT / 2 - 20; y < V_HEIGHT / 2 + 20; y++) {
				cv::Vec3b pixel = frameHSV.at<cv::Vec3b>(y, x);
				if (pixel[0] < minH) { minH = pixel[0]; }
				if (pixel[0] > maxH) { maxH = pixel[0]; }
				if (pixel[0] < minS) { minH = pixel[1]; }
				if (pixel[0] > maxS) { minH = pixel[1]; }
				if (pixel[0] < minV) { minH = pixel[2]; }
				if (pixel[0] > maxV) { minH = pixel[2]; }
				//				pixelsH.push_back(pixel[0]);
				//				pixelsS.push_back(pixel[1]);
				//				pixelsV.push_back(pixel[2]);
			}
		}
		rectangle(currentFrame, Point(V_WIDTH / 2 - 20, V_HEIGHT / 2 - 20), Point(V_WIDTH / 2 + 20, V_HEIGHT / 2 + 20), CV_RGB(255, 0, 0), 1);
		//		inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameThreshold);

		//			putText(currentFrame, "Umiesc obiekt na krzyzyku. Probka " + intToString(pixelsH.size()) + "/200", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
		line(currentFrame, Point(V_WIDTH / 2 - 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 + 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);
		line(currentFrame, Point(V_WIDTH / 2 + 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 - 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);




		//		}
		//		else {
		//			putText(currentFrame, "OK, pobrano probke", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);

		H_MIN = minH;// mean(pixelsH)[0];// -20;
					 //	if (H_MIN < 0) H_MIN = 0;
		S_MIN = minS;// mean(pixelsS)[0];// -20;
					 //	if (S_MIN < 0) S_MIN = 0;
		V_MIN = minV;// mean(pixelsV)[0];// -20;
					 //	if (V_MIN < 0) V_MIN = 0;
		H_MAX = maxH;// mean(pixelsH)[0];// +20;
					 //	if (H_MAX > 255) H_MAX = 255;
		S_MAX = maxS;// mean(pixelsS)[0];// +20;
					 //	if (S_MAX > 255) S_MAX = 255;
		V_MAX = maxV;// mean(pixelsV)[0];// +20;
					 //	if (V_MAX > 255) V_MAX = 255;

					 /*		std::cout << H_MIN << std::endl;
					 std::cout << S_MIN << std::endl;
					 std::cout << V_MIN << std::endl;
					 std::cout << H_MAX << std::endl;
					 std::cout << S_MAX << std::endl;
					 std::cout << V_MAX << std::endl;
					 */

		setTrackbarPos("H_MIN", optionsWindow, H_MIN);
		setTrackbarPos("S_MIN", optionsWindow, S_MIN);
		setTrackbarPos("V_MIN", optionsWindow, V_MIN);
		setTrackbarPos("H_MAX", optionsWindow, H_MAX);
		setTrackbarPos("S_MAX", optionsWindow, S_MAX);
		setTrackbarPos("V_MAX", optionsWindow, V_MAX);

		inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameThreshold);
		imshow("test", frameThreshold);
		//		isObjectDetected = true;
		pixelsH.clear();
		pixelsS.clear();
		pixelsV.clear();

		//	}
		//	else {
		/*	inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameThreshold);
		imshow("test", frameThreshold);

		Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
		//dilate with larger element so make sure object is nicely visible
		Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
		*/

	}

}

void handTracking() {
	/*	Mat hsv, threshold;
	cvtColor(currentFrame, hsv, COLOR_BGR2HSV);
	Scalar lowerSkin = Scalar(0, 36, 83);
	Scalar upperSkin = Scalar(179, 151, 228);

	//	inRange(hsv, lowerSkin, upperSkin, threshold);
	inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);

	///*
	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(threshold, threshold, erodeElement);
	dilate(threshold, threshold, dilateElement);
	erode(threshold, threshold, erodeElement);
	dilate(threshold, threshold, dilateElement);



	//	/*
	erodeElement = getStructuringElement(MORPH_RECT, Size(6, 6));
	dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
	erode(threshold, threshold, erodeElement);
	dilate(threshold, threshold, dilateElement);
	*/
	Mat temp, threshold = applyMask();
	std::vector< std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	threshold.copyTo(temp);
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	bool objectFound = false;
	int numObjects = hierarchy.size();
	Moments max;
	double maxArea = 0;
	int x, y;
	int MIN_OBJECT_AREA = 20 * 20;
	int MAX_OBJECT_AREA = 50 * 50;
	Point2f c, maxC;
	float r;
	float maxR = 0;
	vector<Point> maxContours;

	if (numObjects > 0) {
		// /*
		for (int i = 0; i >= 0; i = hierarchy[i][0]) {
			minEnclosingCircle(contours[i], c, r);
			if (r > maxR) {
				maxR = r;
				maxContours = contours[i];
				maxC = c;
			}
		}
		if ( maxR > 50 ) {
			detectFingest(maxContours, maxC, maxR, threshold);
		}
		// *//
		/*
		for (int index = 0; index >= 0; index = hierarchy[index][0]) {
		Moments moment = moments((cv::Mat)contours[index]);
		double area = moment.m00;
		if (maxArea < area) {
		maxArea = area;
		max = moment;
		}
		}

		double area = max.m00;

		if (area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA) {
		x = max.m10 / area;
		y = max.m01 / area;
		circle(currentFrame, Point(x, y), 30, Scalar(0, 255, 0), 1);
		}
		*/
	}
	imshow("prev", threshold);
}

Mat applyMask() {
	Mat HSV, threshold;
	cvtColor(currentFrame, HSV, COLOR_BGR2HSV);
	inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(threshold, threshold, erodeElement);



	dilate(threshold, threshold, dilateElement);
	erode(threshold, threshold, erodeElement);
	dilate(threshold, threshold, dilateElement);
	return threshold;
}

void detectFingerMoments() {

}

void detectFingest(vector<Point> contours, Point2f c, float r, Mat &threshold) {
	//	cv::minEnclosingCircle(contours, c, r);
	// srodek 0r
	Vec3b lastColor, currentColor;
	Scalar blue = Scalar(255, 0, 0);
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);

	int objectsPerLine = 0;
	//	for (int x = c.x - r; x < c.x + r; x++) {
	//		currentColor = threshold.at<cv::Vec3b>(c.y, x);
	//		if (lastColor[0] == 0.0 && currentColor[0] == 255.0) {
	//			objectsPerLine++;
	//		}
	//		lastColor = currentColor;
	//	}
	//	if (objectsPerLine > 2) {
	//		circle(currentFrame, c, r, green, 3);		
	//	}
	//	objectsPerLine = 0;
	// nadgarstek -0.5r
	//	for (int x = c.x - r; x < c.x + r; x++) {
	//		currentColor = threshold.at<cv::Vec3b>(c.y - r / 2, x);
	//		if (lastColor[0] == 0.0 && currentColor[0] == 255.0) {
	//			objectsPerLine++;
	//		}
	//		lastColor = currentColor;
	//	}
	//	if (objectsPerLine > 1) {
	//		circle(currentFrame, c, r, blue, 3);
	//		return;
	//	}
	// palce 0.5r
	vector<Point> fingers;
	objectsPerLine = 0;
	Point overObject = Point(0, 0);
/*	for (int y = c.y - r; y < c.y; y++) {
		for (int x = c.x - r; x < c.x + r; x++) {
		
			currentColor = threshold.at<cv::Vec3b>(c.y - 0.75 * r, x);
			if (lastColor[0] == 0.0 && currentColor[0] == 255.0) {
				objectsPerLine++;
				overObject = Point(x, y);
				//circle(threshold, Point(x, y), 1, blue, 1);
			}
			if (lastColor[0] == 255.0 && currentColor[0] == 0.0 && overObject.x > 0) {
				//			std::cout << x <<" "<< y << " " << overObject << endl;
				circle(currentFrame, Point( x - abs(x - overObject.x)/2, y), 1, blue, 3);
				//circle(currentFrame, Point(x, y), 1, red, 1);
				objectsPerLine++;
				overObject = Point(0, 0);
			}
			lastColor = currentColor;
			objectsPerLine = 0;
		}
		overObject = Point(0, 0);
		lastColor = Vec3b(0, 0, 0);
	}*/
	int y = c.y - 0.5*r;
	for (int x = c.x - r; x < c.x + r; x++) {

		currentColor = threshold.at<cv::Vec3b>( y, x);
	//	cout << currentColor << endl;
			if (lastColor[0] == 0 && currentColor[0] > 0) {
				circle(currentFrame, Point(x,y), 2, blue);
				objectsPerLine++;
			}
			lastColor = currentColor;
	}
	circle(currentFrame, c, r, red, 3);
	//	line(currentFrame, Point(c.x - r, c.y - r / 2), Point(c.x * r, c.y - r / 2), blue);
	std::cout << "Wykrylem " << objectsPerLine << " palcow" << endl;
}

void detectContours()
{
	Mat temp, frameGray, frameHSV;
	cvtColor(currentFrame, frameGray, CV_BGR2GRAY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	threshold(frameGray, frameGray, thresholdValue, 255, thresholdType);

	Canny(frameGray, frameGray, thresholdValue, 255, 3);

	findContours(frameGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	temp = Mat::zeros(frameGray.size(), CV_8UC3);

	RNG rng(12345);
	for (int i = 0; i< contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(temp, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	currentFrame = temp;
}


void createPreviewWindow() {
	namedWindow(previewWindow, CV_WINDOW_NORMAL);
}

void createOptionsWindow() {
	namedWindow(optionsWindow, CV_WINDOW_NORMAL);

	char TrackbarName[50];
	sprintf_s(TrackbarName, "H_MIN", H_MIN);
	sprintf_s(TrackbarName, "H_MAX", H_MAX);
	sprintf_s(TrackbarName, "S_MIN", S_MIN);
	sprintf_s(TrackbarName, "S_MAX", S_MAX);
	sprintf_s(TrackbarName, "V_MIN", V_MIN);
	sprintf_s(TrackbarName, "V_MAX", V_MAX);

	createTrackbar("Opcja", optionsWindow, &selectedOption, 5, optionChanged);
	createTrackbar("Threshold", optionsWindow, &thresholdValue, 255);
	createTrackbar("ThresholdType", optionsWindow, &thresholdType, 4);
	createTrackbar("H_MIN", optionsWindow, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", optionsWindow, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", optionsWindow, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", optionsWindow, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", optionsWindow, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", optionsWindow, &V_MAX, V_MAX, on_trackbar);

	Scalar lowerSkin = Scalar(0, 61, 81);
	Scalar upperSkin = Scalar(202, 255, 255);
	setTrackbarPos("H_MIN", optionsWindow, lowerSkin[0]);
	setTrackbarPos("S_MIN", optionsWindow, lowerSkin[1]);
	setTrackbarPos("V_MIN", optionsWindow, lowerSkin[2]);
	setTrackbarPos("H_MAX", optionsWindow, upperSkin[0]);
	setTrackbarPos("S_MAX", optionsWindow, upperSkin[1]);
	setTrackbarPos("V_MAX", optionsWindow, upperSkin[2]);

}

void faceDetect() {
	Mat frameGray, foundFace;
	vector<Rect> found;
	cvtColor(currentFrame, frameGray, CV_BGR2GRAY);

	haar_cascade.detectMultiScale(frameGray, found);
	for (int i = 0; i < found.size(); i++) {
		Rect face_i = found[i];
		rectangle(currentFrame, face_i, CV_RGB(255, 0, 0), 1);
	}
}


void init() {
	haar_cascade.load("../data/haarcascade_frontalface_alt.xml");
	createOptionsWindow();
	createPreviewWindow();
}

void runVideo() {
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cerr << "Nie udalo sie otworzyc strumienia video" << endl;
		exit(EXIT_FAILURE);
	}

	capture.set(CV_CAP_PROP_FRAME_WIDTH, V_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, V_HEIGHT);

	while (true) {
		capture >> currentFrame;
		flip(currentFrame, currentFrame, 1);
		handTracking();
		applyOption();
		imshow(previewWindow, currentFrame);
		if (waitKey(33) == 1048603) break;
	}
}

int main(int argc, char *argv[]) {

	init();
	runVideo();

	return EXIT_SUCCESS;

}
