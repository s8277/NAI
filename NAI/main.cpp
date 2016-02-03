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



using namespace cv;

String optionsWindow = "optionsWindow";
String previewWindow = "previewWindow";

CascadeClassifier haar_cascade;
Mat currentFrame;
int selectedOption = 0;
int thresholdValue = 127;
int contourSelected = 4;

	int max_thresh = 255;
	RNG rng(12345);

void optionChanged(int status, void* data);
void thresh_callback(int, void*);
void applyOption();
void faceDetect();


void applyOption() {
	switch (selectedOption) {
	case 1:	
		cvtColor(currentFrame, currentFrame, CV_BGR2GRAY);
		break;
	case 2:
		cvtColor(currentFrame, currentFrame, CV_BGR2HSV);

		break;
	case 3:
		threshold(currentFrame, currentFrame, thresholdValue, 255, 0);
		break;
	case 4:
		thresh_callback(0,0);
		break;
	case 5:
		faceDetect();
		break;
	}

}

void thresh_callback(int, void*)
{
	Mat frameCanny, frameGray;
	cvtColor(currentFrame, frameGray, CV_BGR2GRAY);
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(frameGray, frameCanny, thresholdValue, thresholdValue * 2, 3);
	/// Find contours
	findContours(frameCanny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(frameCanny.size(), CV_8UC3);

	
//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//	drawContours(drawing, contours, contourSelected, color, 2, 8, hierarchy, 0, Point());
	for (int i = 0; i< contours.size(); i++)
	{
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Scalar color( 255, 255, 255);
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
	namedWindow(optionsWindow, CV_WINDOW_AUTOSIZE);

	createTrackbar("Option", optionsWindow, &selectedOption, 7);//, optionChanged);

	//switch (selectedOption) {
	//	case 3:
	//	case 4:
			createTrackbar( "Threshold", optionsWindow, &thresholdValue, 255);
			
	//		break;
	//}
	
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
	
	destroyWindow(optionsWindow);
	createOptionsWindow();
}

int main(int argc, char *argv[]) {
	
	haar_cascade.load("../data/haarcascade_frontalface_alt.xml");
	

	createOptionsWindow();
	createPreviewWindow();

	VideoCapture input(0);


	while (true) {
		input >> currentFrame;
		applyOption();
		imshow(previewWindow, currentFrame);
		int k = waitKey(33);
		if (k >= 1048603) break;
	}



	return 0;
}

// */