#ifndef __KMEANS__
#define __KMEANS__

#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;

namespace kmeans{
	//src画像をk個の領域にクラスタリングする。クラスタリング時に、ピクセルの座標値の値をscale倍して加味する。
	//scale==0の場合、色だけをクラスタリングする。
	void execute(cv::Mat &src, cv::Mat &dst, int k, double scale);
}

#endif