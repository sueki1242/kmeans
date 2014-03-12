#include "kmeans.h"

//src�摜��rgb+xy��5�����x�N�g���Ƃ���5*(�s�N�Z����)pixels�Ɋi�[����B
//xy�̍��W�l�̓p�����[�^scale�ɂ���ăX�P�[�����O�����B
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

//�N���X�^�����O�̌��ʁApixels�̊e�v�f���ǂ̃N���X�^�ɑ����Ă��邩��clusters����擾���A
//�Ή����錳�摜src�̐F��񂩂�N���X�^���Ƃ̕��ϐF�����߂�cluster_colors�ɕԂ��B
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

//�edst�̊e�s�N�Z���ɁA���̃s�N�Z����������N���X�^�̐F�����蓖�Ă�B
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