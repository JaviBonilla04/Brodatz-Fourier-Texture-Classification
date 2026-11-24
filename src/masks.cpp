#include "masks.hpp"
#include <cmath>
#include <limits>


// cl masks.cpp /EHsc /std:c++17 /I "%OPENCV_DIR%\include" /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4130.lib


std::pair<double,double> mask_center(int side){
	return {(double)(side-1)/2, (double)(side-1)/2};
}

std::vector<double> calc_radius(int side, int n_regions) {
	std::vector<double> r = {0.0};
	for(int k = 1; k < n_regions; k++){
		r.push_back(side*sqrt((double)k/(n_regions*CV_PI)));
	}
	r.push_back(std::numeric_limits<double>::infinity());
	return r;
}

static double distance_to_center(const std::pair<double,double> &center, const std::pair<double,double> &pixel){
	return sqrt((center.first - pixel.first)*(center.first - pixel.first) + (center.second - pixel.second)*(center.second - pixel.second));
}

static bool in_region(const std::pair<double,double> &center,const double lower_b, const double radius, const std::pair<double,double> &pixel){
	double dist = distance_to_center(center, pixel);
	if(lower_b<=dist && dist < radius) return true;
	return false;
}

std::vector<cv::Mat> create_masks(const std::pair<double,double> &center, const std::vector<double>& radius, int side){
	std::vector<cv::Mat> masks;
	for(int k = 1;  k < radius.size(); k++) {
		cv::Mat mask(side, side, CV_8UC1, cv::Scalar(0)); 
		for(int i = 0; i < side; i++){	
			for(int j = 0; j < side; j++){

				if(in_region(center, radius[k-1], radius[k], {(double)i, (double)j})) {
					mask.at<uchar>(i,j) = 255;
				}	

			}
		}
		masks.push_back(mask);
	}
	return masks;
}