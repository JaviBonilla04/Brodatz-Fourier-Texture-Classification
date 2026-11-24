#include <filesystem>  // std::filesystem::path, create_directories
#include <fstream>     // std::ofstream
#include <iostream>    // cout, cerr
#include <string>      
#include <vector>     
#include <iomanip>

#include <opencv2/opencv.hpp>

using namespace std;

/*
cl extract_features.cpp /EHsc /std:c++17 /I "%OPENCV_DIR%\include" /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4130.lib

*/


// -- Hyperparameters
const double SIDE = 64;	

cv::Mat power_spectrum(const vector<cv::Mat> &comps){
	cv::Mat power_spec;
	cv::add(comps[0].mul(comps[0]), comps[1].mul(comps[1]), power_spec);
	return power_spec;
}

pair<double,double> calc_center(const int cols, const int rows){
	return {(double)cols/2, (double)rows/2};
}

void center_spectrum(const cv::Mat &img, const pair<double,double> &center){
	double cx = center.first, cy = center.second;

	cv::Mat q_0(img, cv::Rect(0, 0, cx, cy));
	cv::Mat q_1(img, cv::Rect(cx, 0, cx, cy));
	cv::Mat q_2(img, cv::Rect(0, cy, cx, cy));
	cv::Mat q_3(img, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;
	q_0.copyTo(tmp); q_3.copyTo(q_0); tmp.copyTo(q_3);
	q_1.copyTo(tmp); q_2.copyTo(q_1); tmp.copyTo(q_2);
}

cv::Mat create_power_spectrum(const cv::Mat &img_raw, const pair<double,double> &center){
	cv::Mat img;
	img_raw.convertTo(img, CV_32F, 1.0, -cv::mean(img_raw)[0]);

	cv::Mat img_spec;
	cv::dft(img, img_spec, cv::DFT_COMPLEX_OUTPUT);
	
	vector<cv::Mat> comps;
	cv::split(img_spec, comps);
	cv::Mat power_spec = power_spectrum(comps);
	
	center_spectrum(power_spec, center);	
	return power_spec;
}

double calc_entropy(const cv::Mat &img){
	cv::Mat log_p; 
	cv::log(img + 1e-12, log_p);
	double entropy = -(double)cv::sum(img.mul(log_p))[0];
	return entropy;
}

double calc_concentration(const cv::Mat &img){
	double concentration = (double)cv::sum(img.mul(img))[0];
	return concentration;
}

double calc_mean_radial_freq(const cv::Mat &img, const pair<double,double> &center){
	double mean_radial_freq = 0.0, cx = center.first, cy = center.second;
	for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			double u = j-cx, v = i-cy, r = sqrt((double)(u*u + v*v))/img.cols;
			mean_radial_freq += img.at<double>(i, j)*r;
		}
	}
	return mean_radial_freq;
}

double calc_std_radial_freq(const cv::Mat &img, double mean_radial_freq, const pair<double,double> &center){
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

vector<double> calc_features(const cv::Mat &img, const pair<double,double> &center){
	cv::Mat img_norm;
	cv::normalize(img, img_norm, 1.0, 0.0, cv::NORM_L1, CV_64F);

	double entropy = calc_entropy(img_norm);
	double concentration = calc_concentration(img_norm);
	double mean_radial_freq = calc_mean_radial_freq(img_norm, center);
	double std_radial_freq = calc_std_radial_freq(img_norm, mean_radial_freq, center);

	return {entropy, concentration, mean_radial_freq, std_radial_freq};
}

cv::Mat make_cosine(int side, double cycles, double offset = 0.0, double amplitude = 1.0) {
    cv::Mat img(side, side, CV_32F);
    for (int i = 0; i < side; i++)
        for (int j = 0; j < side; j++)
            img.at<float>(i, j) = (float)(offset + amplitude * cos(2.0 * CV_PI * cycles * j / side));
    return img;
}

// Ruido uniforme entre 0 y 255
cv::Mat make_noise(int side, int seed = 42) {
    cv::Mat img(side, side, CV_32F);
    cv::RNG rng(seed);
    rng.fill(img, cv::RNG::UNIFORM, 0.0, 255.0);
    return img;
}

// Imagen constante
cv::Mat make_constant(int side, float value) {
    return cv::Mat(side, side, CV_32F, cv::Scalar(value));
}
	

void print_vec(const vector<double>& vec, string label){
	cout << label << "\n";
	for(double d : vec) {
		cout << d << ", ";
	} cout << "\n";
}

int main(){

	pair<double,double>center = calc_center(SIDE, SIDE);

	cv::Mat img_cosine = make_cosine(SIDE, 8), spec_cosine = create_power_spectrum(img_cosine, center);
	cv::Mat img_cosine_2 = make_cosine(SIDE, 8, 100, 50), spec_cosine_2 = create_power_spectrum(img_cosine_2, center);
	cv::Mat img_noise = make_noise(SIDE), spec_noise = create_power_spectrum(img_noise, center);
	cv::Mat img_constant = make_constant(SIDE, 7), spec_constant = create_power_spectrum(img_constant, center);

	cout << fixed << setprecision(4);

	vector<double> feat_img_cosine = calc_features(spec_cosine, center); print_vec(feat_img_cosine, "conseno");
	vector<double> feat_img_cosine_2 = calc_features(spec_cosine_2, center); print_vec(feat_img_cosine_2, "conseno +100");
	vector<double> feat_img_noise = calc_features(spec_noise, center); print_vec(feat_img_noise, "ruido");
	vector<double> feat_img_constant = calc_features(spec_constant, center); print_vec(feat_img_constant, "constante");


}