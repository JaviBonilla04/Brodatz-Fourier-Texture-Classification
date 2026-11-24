#include "features.hpp"
#include <cmath>

/*
cl features.cpp /EHsc /std:c++17 /I "%OPENCV_DIR%\include" /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4130.lib

*/

std::pair<double,double> spectrum_center(int cols, int rows){
	return {(double)cols/2, (double)rows/2};
}

static cv::Mat power_spectrum(const std::vector<cv::Mat> &comps){
	cv::Mat power_spec;
	cv::add(comps[0].mul(comps[0]), comps[1].mul(comps[1]), power_spec);
	return power_spec;
}

static void center_spectrum(const cv::Mat &img, const std::pair<double,double> &center){
	double cx = center.first, cy = center.second;

	cv::Mat q_0(img, cv::Rect(0, 0, cx, cy));
	cv::Mat q_1(img, cv::Rect(cx, 0, cx, cy));
	cv::Mat q_2(img, cv::Rect(0, cy, cx, cy));
	cv::Mat q_3(img, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;
	q_0.copyTo(tmp); q_3.copyTo(q_0); tmp.copyTo(q_3);
	q_1.copyTo(tmp); q_2.copyTo(q_1); tmp.copyTo(q_2);
}

cv::Mat create_power_spectrum(const cv::Mat &img, const std::pair<double,double> &center){
	cv::Mat f;
	img.convertTo(f, CV_32F, 1.0, -cv::mean(img)[0]);

	cv::Mat img_spec;
	cv::dft(f, img_spec, cv::DFT_COMPLEX_OUTPUT);
	
	std::vector<cv::Mat> comps;
	cv::split(img_spec, comps);
	cv::Mat power_spec = power_spectrum(comps);
	
	center_spectrum(power_spec, center);	
	return power_spec;
}

static double calc_entropy(const cv::Mat &img){
	cv::Mat log_p; 
	cv::log(img + 1e-12, log_p);
	double entropy = -(double)cv::sum(img.mul(log_p))[0];
	return entropy;
}

static double calc_concentration(const cv::Mat &img){
	double concentration = (double)cv::sum(img.mul(img))[0];
	return concentration;
}

static double calc_mean_radial_freq(const cv::Mat &img, const std::pair<double,double> &center){
	double mean_radial_freq = 0.0, cx = center.first, cy = center.second;
	for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			double u = j-cx, v = i-cy, r = sqrt((double)(u*u + v*v))/img.cols;
			mean_radial_freq += img.at<double>(i, j)*r;
		}
	}
	return mean_radial_freq;
}

static double calc_std_radial_freq(const cv::Mat &img, double mean_radial_freq, const std::pair<double,double> &center){
	double std_radial_freq = 0.0, cx = center.first, cy = center.second;
	for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			double u = j-cx, v = i-cy, r = sqrt((double)(u*u + v*v))/img.cols;
			std_radial_freq += (img.at<double>(i, j))*(r-mean_radial_freq)*(r-mean_radial_freq);
		}
	}
	std_radial_freq = sqrt(std_radial_freq);
	return std_radial_freq;
}

std::vector<double> calc_features(const cv::Mat &img, const std::pair<double,double> &center){
	cv::Mat img_norm;
	cv::normalize(img, img_norm, 1.0, 0.0, cv::NORM_L1, CV_64F);

	double entropy = calc_entropy(img_norm);
	double concentration = calc_concentration(img_norm);
	double mean_radial_freq = calc_mean_radial_freq(img_norm, center);
	double std_radial_freq = calc_std_radial_freq(img_norm, mean_radial_freq, center);

	return {entropy, concentration, mean_radial_freq, std_radial_freq};
}

cv::Mat prepare_region(const cv::Mat &crop, const cv::Mat &mask){
	cv::Mat f;
	crop.convertTo(f, CV_32F, 1.0, -cv::mean(crop, mask)[0]);

 	cv::Mat prep = cv::Mat::zeros(f.size(), CV_32F);
 	f.copyTo(prep, mask);
 	return prep;
}

std::vector<double> extract_all(const cv::Mat &crop, const std::vector<cv::Mat> &masks, const std::pair<double,double> &center){
	std::vector<double> descriptor;
	
	cv::Mat global_spec = create_power_spectrum(crop, center);
	std::vector<double> global_desc = calc_features(global_spec, center);
	descriptor.insert(descriptor.end(), global_desc.begin(), global_desc.end());

	for(const cv::Mat& mask : masks){
		cv::Mat prep = prepare_region(crop, mask);
		cv::Mat prep_spec = create_power_spectrum(prep, center);
		std::vector<double> curr_features = calc_features(prep_spec, center);
		descriptor.insert(descriptor.end(), curr_features.begin(), curr_features.end());
	}

	return descriptor;
}
