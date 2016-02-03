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

// /*

// cele:
// wyciąć dłoń z obrazu
// porównać matchShape do przygotowanych obrazków z palcami 1-5
// w zależności od poziomu dopasowania do palca polazujemy jakąś opcję


using namespace cv;
using namespace std;

String optionsWindow = "optionsWindow";
String previewWindow = "previewWindow";

CascadeClassifier haar_cascade;
Mat currentFrame;
int selectedOption = 0;
int thresholdValue = 100;
int thresholdType = 3;
int contourSelected = 4;

//	int max_thresh = 255;
//	RNG rng(12345);

void optionChanged(int status, void* data);
void thresh_callback(int, void*);
void applyOption();
void faceDetect();
void objectDetect();

const int V_WIDTH = 640;
const int V_HEIGHT = 480;

int H_MIN = 0;// 0; // 77
int H_MAX = 179;//183; //117
int S_MIN = 36;//71; //22
int S_MAX = 255;//176; // 62
int V_MIN = 83;//87; // 55
int V_MAX = 255;//227; // 95
const int MAX_NUM_OBJECTS = 30;
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = 200 * 200;
void drawObject(int x, int y, Mat &frame);
void on_trackbar(int, void*) {}
std::string intToString(int number);
void drawHandPlace();

bool isObjectDetected = false;
void objectDetectMog2();
void skinMask();
void colorTest();
void newHandTracking();

void applyOption() {
	//	objectDetectMog2();
	Mat thresh, erodeElement, dilateElement, mask, hsv;
	Scalar lowerSkin = Scalar(0, 20, 127);
	Scalar upperSkin = Scalar(23, 113, 255);

	switch (selectedOption) {
	case 1:
		cvtColor(currentFrame, currentFrame, CV_BGR2HSV);
		//threshold(currentFrame, currentFrame, thresholdValue, 255, thresholdType);
		inRange(currentFrame, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), currentFrame);
		//cvtColor(currentFrame, currentFrame, CV_BGR2GRAY);
		//		threshold(currentFrame, currentFrame, thresholdValue, 255, thresholdType);
		break;
	case 2:
		cvtColor(currentFrame, currentFrame, CV_BGR2HSV_FULL);
		inRange(currentFrame, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), currentFrame);
		//		erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));		
		//		dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
		//		erode(currentFrame, currentFrame, erodeElement);
		//		dilate(currentFrame, currentFrame, dilateElement);


		break;
	case 3: // threshold
		threshold(currentFrame, currentFrame, thresholdValue, 255, thresholdType);
		cvtColor(currentFrame, currentFrame, CV_BGR2HSV_FULL);
				inRange(currentFrame, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), currentFrame);
		break;
	case 4: // kontury
			//	thresh_callback(0, 0);
		break;
	case 5:
		objectDetect();
		break;
	case 6:
		objectDetectMog2();
		break;
	case 7:
		newHandTracking();
		break;
	case 8:
		faceDetect();
		break;
	case 9:
		skinMask();
		break;
	default:
		//	colorTest();
		newHandTracking();
		//drawHandPlace();
		break;
	}

}

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
					currentColor = hsv.at<cv::Vec3b>(c.y - r/2, x);
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

		putText(currentFrame, intToString(pixel[0]) + "," + intToString(pixel[1]) + "," + intToString(pixel[2]), Point(handPoints[i].x - 10, handPoints[i].y + 20), 2, 1, pixel, 1, 4);



		//	line(currentFrame, Point(handPoints[i].x, handPoints[i].y), Point(V_WIDTH / 2 + 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);
	}

	//	line(currentFrame, Point(V_WIDTH / 2 - 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 + 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);
	//	line(currentFrame, Point(V_WIDTH / 2 + 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 - 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);

}

void objectDetect() {
	Mat frameHSV;
	Mat frameThreshold;

	threshold(currentFrame, currentFrame, 180, 255, thresholdType);

	cvtColor(currentFrame, frameHSV, COLOR_BGR2HSV);

	if (!isObjectDetected) {

		if (pixelsH.size() < 200) {

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



			cv::Vec3b pixel = frameHSV.at<cv::Vec3b>(V_WIDTH / 2, V_HEIGHT / 2);

			pixelsH.push_back(pixel[0]);
			pixelsS.push_back(pixel[1]);
			pixelsV.push_back(pixel[2]);
			inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameThreshold);

			putText(currentFrame, "Umiesc obiekt na krzyzyku. Probka " + intToString(pixelsH.size()) + "/200", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
			line(currentFrame, Point(V_WIDTH / 2 - 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 + 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);
			line(currentFrame, Point(V_WIDTH / 2 + 30, V_WIDTH / 2 - 30), Point(V_WIDTH / 2 - 30, V_WIDTH / 2 + 30), Scalar(0, 255, 0), 2);




		}
		else {
			putText(currentFrame, "OK, pobrano probke", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);

			H_MIN = mean(pixelsH)[0] - 20;
			//	if (H_MIN < 0) H_MIN = 0;
			S_MIN = mean(pixelsS)[0] - 20;
			//	if (S_MIN < 0) S_MIN = 0;
			V_MIN = mean(pixelsV)[0] - 20;
			//	if (V_MIN < 0) V_MIN = 0;
			H_MAX = mean(pixelsH)[0] + 20;
			//	if (H_MAX > 255) H_MAX = 255;
			S_MAX = mean(pixelsS)[0] + 20;
			//	if (S_MAX > 255) S_MAX = 255;
			V_MAX = mean(pixelsV)[0] + 20;
			//	if (V_MAX > 255) V_MAX = 255;

			std::cout << H_MIN << std::endl;
			std::cout << S_MIN << std::endl;
			std::cout << V_MIN << std::endl;
			std::cout << H_MAX << std::endl;
			std::cout << S_MAX << std::endl;
			std::cout << V_MAX << std::endl;


			setTrackbarPos("H_MIN", optionsWindow, H_MIN);
			setTrackbarPos("S_MIN", optionsWindow, S_MIN);
			setTrackbarPos("V_MIN", optionsWindow, V_MIN);
			setTrackbarPos("H_MAX", optionsWindow, H_MAX);
			setTrackbarPos("S_MAX", optionsWindow, S_MAX);
			setTrackbarPos("V_MAX", optionsWindow, V_MAX);

			inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameThreshold);
			imshow("test", frameThreshold);
			isObjectDetected = true;
			pixelsH.clear();
			pixelsS.clear();
			pixelsV.clear();
		}
	}
	else {
		inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameThreshold);
		imshow("test", frameThreshold);

		Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
		//dilate with larger element so make sure object is nicely visible
		Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));


	}
	/*

	Mat frameHSV;
	Mat threshold;
	cvtColor(currentFrame, frameHSV, COLOR_BGR2HSV);
	inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(threshold, threshold, erodeElement);
	erode(threshold, threshold, erodeElement);


	dilate(threshold, threshold, dilateElement);
	dilate(threshold, threshold, dilateElement);

	//	threshold.copyTo(currentFrame);

	std::vector< std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	findContours(threshold, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	double refArea = 0;
	bool objectFound = false;
	int x, y;
	if ( hierarchy.size() > 0) {

	int numObjects = hierarchy.size();
	//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
	if (numObjects < MAX_NUM_OBJECTS) {
	for (int i = 0; i >= 0; i = hierarchy[i][0]) {
	Moments moment = moments((cv::Mat)contours[i]);
	double area = moment.m00;

	if (area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
	x = moment.m10 / area;
	y = moment.m01 / area;
	objectFound = true;
	refArea = area;
	}
	else objectFound = false;

	}

	if (objectFound == true) {
	putText(currentFrame, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
	//draw object location on screen
	drawObject(x, y, currentFrame);
	}

	}

	}
	*/
	//threshold.copyTo(currentFrame);

}

std::string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}

void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25>0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25<500)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 500), Scalar(0, 255, 0), 2);
	if (x - 25>0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25<500)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(500, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}

void thresh_callback(int, void*)
{
	Mat frameCanny, frameGray, frameHSV;
	//	cvtColor(currentFrame, frameGray, CV_BGR2GRAY);

	//	cvtColor(currentFrame, frameHSV, COLOR_BGR2HSV);

	//	inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameGray);
	//	imshow("hsv", frameGray);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	threshold(frameGray, frameGray, thresholdValue, 255, thresholdType);
	/// Detect edges using canny
	Canny(frameGray, frameGray, thresholdValue, thresholdValue * 2, 3);
	/// Find contours
	findContours(frameGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(frameGray.size(), CV_8UC3);

	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	drawContours(drawing, contours, contourSelected, color, 2, 8, hierarchy, 0, Point());
	for (int i = 0; i< contours.size(); i++)
	{
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Scalar color(255, 255, 255);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	drawing.copyTo(currentFrame);
	/// Show in a window
	//	namedWindow("Contours", WINDOW_AUTOSIZE);
	//	imshow("Contours", drawing);
}


void createPreviewWindow() {
	namedWindow(previewWindow, CV_WINDOW_AUTOSIZE);
}

void createOptionsWindow() {
	namedWindow(optionsWindow, CV_WINDOW_NORMAL);


	createTrackbar("Option", optionsWindow, &selectedOption, 7, optionChanged);

	//	switch (selectedOption) {
	//		case 3:
	//		case 4:
	createTrackbar("Threshold", optionsWindow, &thresholdValue, 255);
	createTrackbar("ThresholdType", optionsWindow, &thresholdType, 4);
	//	break;
	//		case 5:
	//		case 6:
	//		case 7:
	// /*
	createTrackbar("H_MIN", optionsWindow, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", optionsWindow, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", optionsWindow, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", optionsWindow, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", optionsWindow, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", optionsWindow, &V_MAX, V_MAX, on_trackbar);
	// */
	//			break;
	//	}

}

void faceDetect() {


	Mat prev;

	//	cvtColor(currentFrame, gray, CV_BGR2GRAY);
	// Find the faces in the frame:
	//	std::vector< Rect_<int> > faces;

	int groundThreshold = 2;
	double scaleStep = 1.1;
	Size minimalObjectSize(80, 80);
	Size maximalObjectSize(300, 300);
	std::vector<Rect> found;
	currentFrame.copyTo(prev);
	//	haar_cascade.detectMultiScale(gray, faces);
	Mat image_grey;

	cvtColor(currentFrame, image_grey, CV_BGR2GRAY);

	// Detect faces
	haar_cascade.detectMultiScale(image_grey, found);// , scaleStep, groundThreshold, 0, minimalObjectSize, maximalObjectSize);
													 /*
													 if (found.size() > 0) {
													 for (int i = 0; i <= 2; i++) {
													 rectangle(prev, found[i].br(), found[i].tl(), Scalar(0, 0, 0), 1, 8, 0);

													 }
													 }
													 // */
													 // /*
	for (int i = 0; i < found.size(); i++) {
		// Process face by face:
		Rect face_i = found[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = image_grey(face_i);
		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:
		Mat face_resized;
		cv::resize(face, face_resized, Size(image_grey.cols, image_grey.rows), 1.0, 1.0, INTER_CUBIC);

		// And finally write all we've found out to the original image!
		// First of all draw a green rectangle around the detected face:
		rectangle(currentFrame, face_i, CV_RGB(0, 255, 0), 1);

	}
	// */

	//	imshow("wooohooo", prev);



}

void optionChanged(int status, void* data) {
	isObjectDetected = false;
	//	destroyWindow(optionsWindow);
	//	createOptionsWindow();
}

int main(int argc, char *argv[]) {

	haar_cascade.load("../data/haarcascade_frontalface_alt.xml");


	createOptionsWindow();
	createPreviewWindow();

	VideoCapture input(0);

	//	if (!capture)
	//	{
	//		printf("!!! Failed cvCaptureFromCAM\n");
	//		return 1;
	//	}

	input.set(CV_CAP_PROP_FRAME_WIDTH, V_WIDTH);
	input.set(CV_CAP_PROP_FRAME_HEIGHT, V_HEIGHT);

	while (true) {
		input >> currentFrame;
		flip(currentFrame, currentFrame, 1);
		applyOption();
		imshow(previewWindow, currentFrame);
		int k = waitKey(33);
		if (k >= 1048603) break;
	}



	return 0;
}
