#ifndef MASKS_HPP
#define MASKS_HPP

#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

std::pair<double,double> mask_center(int side);

std::vector<double> calc_radius(int side, int n_regions);

std::vector<cv::Mat> create_masks(
	const std::pair<double,double> &center,
	const std::vector<double> &radius,
	int side);


#endif