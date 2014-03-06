#ifndef __KMEANS__
#define __KMEANS__

#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;

namespace kmeans{
	//src画像をk個の領域にクラスタリングする。
	void execute(cv::Mat &src, cv::Mat &dst, int k);
}

#endif