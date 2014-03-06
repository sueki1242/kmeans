#include "kmeans.h"

void kmeans::execute(cv::Mat &src, cv::Mat &dst, int k){
	//create feature vectors
	cv::Mat pixels;
	src.convertTo(pixels, CV_32FC3);
	pixels = pixels.reshape(1, src.cols*src.rows);

	//clustering
	cv::Mat_<int> clusters(pixels.size(), CV_32SC1);
	cv::Mat centers(cv::Size(1,k), CV_32FC3);
	cv::kmeans(pixels, k, clusters, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);

	//assign clustering result to dst
	cv::MatIterator_<cv::Vec3f> itf = centers.begin<cv::Vec3f>();
	cv::MatIterator_<cv::Vec3b> itd = dst.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> itd_end = dst.end<cv::Vec3b>();
	for (int i = 0; itd != itd_end; ++itd, ++i){
		cv::Vec3f color = itf[clusters.at<int>(1, i)];
		(*itd)[0] = cv::saturate_cast<uchar>(color[0]);
		(*itd)[1] = cv::saturate_cast<uchar>(color[1]);
		(*itd)[2] = cv::saturate_cast<uchar>(color[2]);
	}
}