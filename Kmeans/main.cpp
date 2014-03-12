#include <iostream>
using namespace std;

#include "kmeans.h"


int main(){
	string imgPath;
	cin >> imgPath;
	cv::Mat src = cv::imread(imgPath);
	cv::Mat dst = src.clone();
	kmeans::execute(src, dst, 20, 1.6);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::waitKey();

	return 0;
}