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

CascadeClassifier haarCascadeFace, haarCascadeHand;
// current analyzed frame 
//Mat currentFrame, currentTemp;
// current filter
int selectedOption = 0;
// global settings for parameters
int thresholdValue = 100;
int thresholdType = 3;

Ptr<BackgroundSubtractor> mog2;


// function declarations
void optionChanged(int status, void* data) {};
Mat applyOption(Mat currentFrame);
Mat faceDetect(Mat currentFrame);
void on_trackbar(int, void*) {}
void drawHandPlace(Mat currentFrame);
void objectDetectMog2(Mat currentFrame);
Mat skinMask(Mat currentFrame);
void colorTest(Mat currentFrame);
Mat newHandTracking(Mat currentFrame);


const int MAX_NUM_OBJECTS = 30;
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = 200 * 200;
bool isObjectDetected = false;


Mat detectContours(Mat currentFrame);
Mat detectObject(Mat currentFrame);
Mat substractMove(Mat currentFrame);
Mat colorDetect(Mat currentFrame) { return currentFrame; };
void showMenu() {};
Mat handTracking(Mat currentFrame);
Mat detectFingest(vector<Point> contours, Point2f c, float r, Mat threshold, Mat currentFrame);
Mat applyMask(Mat currentFrame);

Mat applyOption(Mat currentFrame) {

	switch (selectedOption) {
	case 1: // object contours
		currentFrame = detectContours(currentFrame);
		break;
	case 2: // skin object
		currentFrame = skinMask(currentFrame);
		break;
	case 3: // movement substraction
		currentFrame = substractMove(currentFrame);
		break;
	case 4: // face detection
		currentFrame = faceDetect(currentFrame);
		break;
//	case 5: // 
//		currentFrame = colorDetect(currentFrame);
//		break;
	default:
		break;
	}
	return currentFrame;
}

Mat skinMask(Mat currentFrame) {
	Mat thresh, erodeElement, dilateElement, mask, hsv;
	Scalar lowerSkin = Scalar(0, 20, 127);
	Scalar upperSkin = Scalar(23, 113, 255);

	//	lowerSkin = Scalar(0, 36, 83);
	//	upperSkin = Scalar(179, 151, 228);

	cvtColor(currentFrame, hsv, COLOR_BGR2HSV_FULL);

	inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), mask);

	erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
	erode(mask, mask, erodeElement);
	dilate(mask, mask, dilateElement);

	bitwise_and(currentFrame, currentFrame, thresh, mask = mask);

	return thresh;
}

Mat detectContours(Mat currentFrame)
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
	return temp;
}

Mat substractMove(Mat currentFrame) {
	mog2->apply(currentFrame, currentFrame);
	return currentFrame;
}


Mat detectFingest(vector<Point> contours, Point2f c, float r, Mat threshold, Mat currentFrame) {
	Vec3b lastColor, currentColor = Vec3b(100,100,100);
	Scalar blue = Scalar(255, 0, 0);
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);

	int objectsPerLine = 0;

// nadgarstek -0.5r
/*
	for (int x = c.x - r; x < c.x + r; x++) {
		currentColor = threshold.at<cv::Vec3b>(c.y - r / 2, x);
		if (lastColor[0] == 0.0 && currentColor[0] == 255.0) {
			objectsPerLine++;
		}
		lastColor = currentColor;
	}
	if (objectsPerLine > 1) {
		circle(currentFrame, c, r, blue, 3);
		return currentFrame;
	}
*/
	vector<Point> fingers;
	objectsPerLine = 0;
	vector<int> line;
	Point overObject = Point(0, 0);

	for (int y = c.y - 0.75 * r; y < c.y - 0.25*r; y += 10) {
		for (int x = c.x - r; x < c.x + r; x++) {
			currentColor = threshold.at<cv::Vec3b>(y, x);
			if (lastColor[0] == 0 && currentColor[0] == 255) {
				overObject = Point(x, y);
			}
			if (lastColor[0] == 255 && currentColor[0] == 0 && overObject.x > 0) {
				objectsPerLine++;
				overObject = Point(0, 0);
			}
			lastColor = currentColor;
circle(currentFrame, Point(x, y), 1, Scalar(0, 0, 255));
		}
		line.push_back(objectsPerLine);

		objectsPerLine = 0;
		overObject = Point(0, 0);
		lastColor = Vec3b(0, 0, 0);
	}
	int fingersCount = floor(mean(line)[0]/3);
	
	if (fingersCount > 0 && fingersCount < 6) {
		putText(currentFrame, to_string(fingersCount), Point(0, 50), 5, 2, Scalar(0, 0, 255), 2);
		selectedOption = fingersCount;
	}

	return currentFrame;
}

Mat handTracking2(Mat currentFrame) {
// /*	
	Mat temp;
	currentFrame.copyTo(temp);
	inRange(temp, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), temp);
	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
	erode(temp, temp, erodeElement);

	dilate(temp, temp, dilateElement);
	erode(temp, temp, erodeElement);
	dilate(temp, temp, dilateElement);

	std::vector< std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	Mat tmp = temp.clone();
//	imshow("Kolor", temp);
	findContours(tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
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
		for (int i = 0; i >= 0; i = hierarchy[i][0]) {
			minEnclosingCircle(contours[i], c, r);
			if (r > maxR) {
				maxR = r;
				maxContours = contours[i];
				maxC = c;
			}
		}
		if (maxR > 150 ) {
			circle(currentFrame, maxC, maxR, Scalar(0, 255, 0), 3);
			currentFrame = detectFingest(maxContours, maxC, maxR, temp, currentFrame);
		}

	}
	return currentFrame;
}

Mat handTracking(Mat currentFrame) {
	return handTracking2(currentFrame);


	Mat frame2;
	RNG rng(12345);
	currentFrame.copyTo(frame2);

	mog2->apply(frame2, frame2, -0.009);

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(2, 2));
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(5, 5));
		
	erode(frame2, frame2, dilateElement);
	dilate(frame2, frame2, dilateElement);
	erode(frame2, frame2, erodeElement);
	dilate(frame2, frame2, dilateElement);

	erodeElement = getStructuringElement(MORPH_RECT, Size(4, 4));
	dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
	erode(frame2, frame2, erodeElement);
	dilate(frame2, frame2, dilateElement);
	inRange(frame2, Scalar(0, 0, 0), Scalar(1, 255, 255), frame2);

	Mat temp, threshold;
	std::vector< std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	bitwise_not(frame2, frame2);

	findContours(frame2.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
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
		for (int i = 0; i >= 0; i = hierarchy[i][0]) {
			minEnclosingCircle(contours[i], c, r);
			if (r > maxR) {
				maxR = r;
				maxContours = contours[i];
				maxC = c;
			}
		}
		if (maxR > 100 && maxR < 200) {
			cout << maxR << endl;
			circle(currentFrame, c, r, Scalar(0, 255, 0), 3);
			currentFrame = detectFingest(maxContours, maxC, maxR, frame2, currentFrame);
		}

	}
	imshow("handTracking", frame2);
	return currentFrame;
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

	Scalar lowerSkin = Scalar(0, 15, 103);
	Scalar upperSkin = Scalar(255, 114, 188);

	lowerSkin = Scalar(62,79,139); //0,61,122
	upperSkin = Scalar(135,237,214);// 105 161 188

	setTrackbarPos("H_MIN", optionsWindow, lowerSkin[0]);
	setTrackbarPos("S_MIN", optionsWindow, lowerSkin[1]);
	setTrackbarPos("V_MIN", optionsWindow, lowerSkin[2]);
	setTrackbarPos("H_MAX", optionsWindow, upperSkin[0]);
	setTrackbarPos("S_MAX", optionsWindow, upperSkin[1]);
	setTrackbarPos("V_MAX", optionsWindow, upperSkin[2]);

}

Mat faceDetect(Mat currentFrame) {
	Mat frameGray, foundFace;
	vector<Rect> found;
	cvtColor(currentFrame, frameGray, CV_BGR2GRAY);

	haarCascadeFace.detectMultiScale(frameGray, found);
	for (int i = 0; i < found.size(); i++) {
		Rect face_i = found[i];
		rectangle(currentFrame, face_i, CV_RGB(255, 0, 0), 1);
	}
	return currentFrame;
}


void init() {
	haarCascadeFace.load("../data/haarcascade_frontalface_alt.xml");
//	haarCascadeHand.load("../data/haarcascade_hand.xml");
	mog2 = createBackgroundSubtractorKNN();
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
	Mat currentFrame, temp;
	while (true) {
		capture >> currentFrame;
		flip(currentFrame, currentFrame, 1);
		temp = handTracking(currentFrame.clone());
		currentFrame = applyOption(currentFrame.clone());
		imshow(previewWindow, currentFrame);
		imshow("Palce", temp);
		if (waitKey(33) == 1048603) break;
	}
}

int main(int argc, char *argv[]) {

	init();
	runVideo();

	return EXIT_SUCCESS;

}


///////////////////////////////////////
/*
std::vector< Vec3b > pixelsH;
std::vector< Vec3b > pixelsS;
std::vector< Vec3b > pixelsV;

void colorTest( Mat currentFrame) {
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
	imshow("hlsf", hlsf);* /
}
*/
/*
Mat newHandTracking( Mat currentFrame) {
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
			
		}

	}

	return currentFrame;


}
*/
/*
bool handFound = false;
Scalar skinMin = Scalar();
Scalar skinMax = Scalar();

void drawHandPlace(Mat currentFrame) {
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
	* /
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
*/
/*
void drawHandPlace2(Mat currentFrame) {
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

*/
/*
Mat detectObject(Mat currentFrame) {
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
		* /


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
					 * /

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
		* /

	}
	return currentFrame;
}
*/
/*
Mat applyMask(Mat currentFrame) {
	Mat HSV, threshold;
	cvtColor(currentFrame, HSV, COLOR_BGR2HSV);
	inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);


	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(threshold, threshold, erodeElement);
	dilate(threshold, threshold, dilateElement);
	erode(threshold, threshold, erodeElement);
	dilate(threshold, threshold, dilateElement);

	erodeElement = getStructuringElement(MORPH_RECT, Size(6, 6));
	dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
	erode(threshold, threshold, erodeElement);
	dilate(threshold, threshold, dilateElement);

	return threshold;
}

*/




