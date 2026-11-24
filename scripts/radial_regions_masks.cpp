#include <filesystem>  // std::filesystem::path, create_directories
#include <fstream>     // std::ofstream
#include <iostream>    // cout, cerr
#include <cmath>
#include <string>      
#include <vector>    
#include <limits> 

#include <opencv2/opencv.hpp>

using namespace std;


/*
cl radial_regions_masks.cpp /EHsc /std:c++17 /I "%OPENCV_DIR%\include" /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4130.lib

*/


// -- System Data
const string FOLDER = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\results\\figures\\masks\\";

// -- Datos 
const int SIDE = 64;
const int N_REGIONS = 3;

void save_mask(const string region, cv::Mat img){
	string route = FOLDER + "mask_"+ region +".png";
	if(!cv::imwrite(route, img)) {
		std::cerr << "Could not save: " << route << "\n";
	}
}

void save_masks(vector<cv::Mat> &masks) {
	std::filesystem::remove_all(FOLDER);
	std::filesystem::create_directories(FOLDER);

	for(int i = 0; i < masks.size(); i++) save_mask("R"+to_string(i+1), masks[i]);	
	cout << "All masks are saved" << "\n";
}

double distance_to_center(const pair<double,double> &center, const pair<double,double> &pixel){
	return sqrt((center.first - pixel.first)*(center.first - pixel.first) + (center.second - pixel.second)*(center.second - pixel.second));
}

bool in_region(const pair<double,double> &center,const double lower_b, const double radius, const pair<double,double> &pixel){
	double dist = distance_to_center(center, pixel);
	if(lower_b<=dist && dist < radius) return true;
	return false;
}

vector<cv::Mat> create_masks(const pair<double,double> &center, const vector<double>& radius, const int side){
	vector<cv::Mat> masks;
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

vector<double> calc_radius(const int side, const int n) {
	vector<double> r = {0.0};
	for(int k = 1; k < n; k++){
		r.push_back(side*sqrt((double)k/(n*CV_PI)));
	}
	r.push_back(numeric_limits<double>::infinity());
	return r;
}

pair<double,double> calc_center(const int side){
	return {(double)(side-1)/2, (double)(side-1)/2};
}

int main(){
	pair<double,double> center = calc_center(SIDE);
	vector<double> radius = calc_radius(SIDE, N_REGIONS);
	vector<cv::Mat> masks = create_masks(center, radius, SIDE);
	save_masks(masks);
}