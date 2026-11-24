#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

std::pair<double,double> spectrum_center(int cols, int rows);

cv::Mat create_power_spectrum(
	const cv::Mat &img, 
	const std::pair<double,double> &center);

std::vector<double> calc_features(
	const cv::Mat &img,
	const std::pair<double,double> &center);

cv::Mat prepare_region(
	const cv::Mat &crop, 
	const cv::Mat &mask);

std::vector<double> extract_all(
	const cv::Mat &crop,
	const std::vector<cv::Mat> &masks,
	const std::pair<double,double> &center);

#endif