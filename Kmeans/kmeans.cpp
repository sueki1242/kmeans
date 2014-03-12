#include "kmeans.h"

//src画像をrgb+xyの5次元ベクトルとして5*(ピクセル数)pixelsに格納する。
//xyの座標値はパラメータscaleによってスケーリングされる。
void calculateFeatureVectors(cv::Mat &src, cv::Mat &pixels, double scale){
	pixels = cv::Mat(cv::Size(5, src.cols*src.rows), CV_32F);
	for (int i = 0, y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++, i++){
			pixels.at<float>(i, 0) = src.at<cv::Vec3f>(y, x)[0];
			pixels.at<float>(i, 1) = src.at<cv::Vec3f>(y, x)[1];
			pixels.at<float>(i, 2) = src.at<cv::Vec3f>(y, x)[2];
			pixels.at<float>(i, 3) = y * scale;
			pixels.at<float>(i, 4) = x * scale;
		}
	}
}

//クラスタリングの結果、pixelsの各要素がどのクラスタに属しているかをclustersから取得し、
//対応する元画像srcの色情報からクラスタごとの平均色を求めてcluster_colorsに返す。
void calculateMeanColor(cv::Mat &pixels, cv::Mat &src, cv::Mat &clusters, vector<cv::Vec3f> &cluster_colors){
	int k = cluster_colors.size();
	vector<int> cluster_sizes(k);

	for (int i = 0, y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++, i++){
			int idx = clusters.at<int>(0, i);
			cluster_sizes[idx]++;
			cluster_colors[idx] += src.at<cv::Vec3f>(y, x);
		}
	}

	for (int i = 0; i < k; i++){
		cluster_colors[i] /= cluster_sizes[i];
	}
}

//各dstの各ピクセルに、そのピクセルが属するクラスタの色を割り当てる。
void assignResult(cv::vector<cv::Vec3f> &cluster_colors, cv::Mat &clusters, cv::Mat &dst){
	cv::MatIterator_<cv::Vec3b> itd = dst.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> itd_end = dst.end<cv::Vec3b>();
	for (int i = 0; itd != itd_end; ++itd, ++i){
		cv::Vec3f color = cluster_colors[clusters.at<int>(0, i)];
		(*itd)[0] = cv::saturate_cast<uchar>(color[0]);
		(*itd)[1] = cv::saturate_cast<uchar>(color[1]);
		(*itd)[2] = cv::saturate_cast<uchar>(color[2]);
	}
}

// usage : dst must be same type and size with src.
void kmeans::execute(cv::Mat &src, cv::Mat &dst, int k, double scale){
	//create feature vectors
	cv::Mat src_f, pixels;
	src.convertTo(src_f, CV_32FC3);
	calculateFeatureVectors(src_f, pixels, scale);

	//clustering
	cv::Mat_<int> clusters(src.rows*src.cols, CV_32SC1);
	cv::Mat centers(cv::Size(5,k), CV_32FC1);
	cv::kmeans(pixels, k, clusters, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);

	//calculare mean color for each region
	vector<cv::Vec3f> cluster_colors(k);
	calculateMeanColor(pixels, src_f, clusters, cluster_colors);

	//assign mean color to dst image
	assignResult(cluster_colors, clusters, dst);
}