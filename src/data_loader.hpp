#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <map>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// id_crop,name_class,row,col,set,route, 16 descriptores
inline constexpr int META_COLS = 6;

struct FeatureSet {
	cv::Mat X;                     // N x 16, CV_32F
	cv::Mat y;                     // N x 1,  CV_32S
	std::vector<std::string> ids;  // id_crop de cada fila, mismo orden que X e y
};

std::map<std::string,int> build_class_map(const std::string &csv_path);

std::vector<std::string> class_names(const std::map<std::string,int> &class_map);

FeatureSet load_features(const std::string &csv_path, const std::map<std::string,int> &class_map);

cv::Mat select_columns(const cv::Mat &X, int start, int end);

#endif