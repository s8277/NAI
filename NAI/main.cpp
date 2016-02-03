threshold demo - ksztalty/kontrast


#include "stdafx.h"

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//#include <string>
//#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

using namespace cv;

/// Global variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo(int, void*);

/**
* @function main
*/
int main(int argc, char** argv)
{
	/// Load an image
	//cv::VideoCapture cap(0);

	src = imread("../data/fruits.jpg", 1);
	cvtColor(src, src_gray, CV_BGR2GRAY);
	/// Create a window to display results
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to choose type of Threshold
	createTrackbar(trackbar_type,
		window_name, &threshold_type,
		max_type, Threshold_Demo);

	createTrackbar(trackbar_value,
		window_name, &threshold_value,
		max_value, Threshold_Demo);

	/// Call the function to initialize
	Threshold_Demo(0, 0);
	
	/// Wait until user finishes program
	while (true)
	{
		//cap >> src;


		/// Convert the image to Gray
		//cvtColor(src, src_gray, CV_BGR2GRAY);


		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}

}


/**
* @function Threshold_Demo
*/
void Threshold_Demo(int, void*)
{
	/* 0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/

	threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);

	imshow(window_name, dst);
}
