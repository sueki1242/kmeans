#ifndef __KMEANS__
#define __KMEANS__

#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;

namespace kmeans{
	//src�摜��k�̗̈�ɃN���X�^�����O����B
	void execute(cv::Mat &src, cv::Mat &dst, int k);
}

#endif