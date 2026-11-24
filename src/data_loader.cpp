#include "data_loader.hpp"
#include "csv_utils.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

static std::vector<std::string> read_data_lines(const std::string &csv_path){
	std::vector<std::string> lines;
	std::ifstream file(csv_path);
	if(!file.is_open()){
		std::cerr << "Could not open: " << csv_path << "\n";
		return lines;
	}
	std::string line;
	std::getline(file, line); // encabezado
	while(std::getline(file, line)){
		if(!line.empty() && line.back() == '\r') line.pop_back();
		if(!line.empty()) lines.push_back(line);
	}
	return lines;
}

std::map<std::string,int> build_class_map(const std::string &csv_path){
	std::vector<std::string> names;
	for(const std::string &line : read_data_lines(csv_path)){
		std::vector<std::string_view> cols = divide_line(line);
		if(cols.size() < 2) continue;
		std::string name(cols[1]);
		if(std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
	}

	// Orden numérico: "D2" antes que "D10" (el orden alfabético los invertiría)
	std::sort(names.begin(), names.end(), [](const std::string &a, const std::string &b){
		return toNum<int>(std::string_view(a).substr(1)) < toNum<int>(std::string_view(b).substr(1));
	});

	std::map<std::string,int> class_map;
	for(int i = 0; i < (int)names.size(); i++) class_map[names[i]] = i;
	return class_map;
}

std::vector<std::string> class_names(const std::map<std::string,int> &class_map){
	std::vector<std::string> names(class_map.size());
	for(const auto &[name, idx] : class_map) names[idx] = name;
	return names;
}

FeatureSet load_features(const std::string &csv_path, const std::map<std::string,int> &class_map){
	FeatureSet data;
	std::vector<std::string> lines = read_data_lines(csv_path);
	if(lines.empty()) return data;

	const int n_feats = (int)divide_line(lines[0]).size() - META_COLS;
	data.X = cv::Mat((int)lines.size(), n_feats, CV_32F);
	data.y = cv::Mat((int)lines.size(), 1, CV_32S);

	int row = 0;
	for(const std::string &line : lines){
		std::vector<std::string_view> cols = divide_line(line);
		if((int)cols.size() != META_COLS + n_feats){
			std::cerr << "Bad number of columns, skipped: " << cols[0] << "\n";
			continue;
		}
		auto it = class_map.find(std::string(cols[1]));
		if(it == class_map.end()){
			std::cerr << "Unknown class, skipped: " << cols[1] << "\n";
			continue;
		}

		for(int j = 0; j < n_feats; j++)
			data.X.at<float>(row, j) = toNum<float>(cols[META_COLS + j]);
		data.y.at<int>(row, 0) = it->second;
		data.ids.emplace_back(cols[0]);
		row++;
	}

	// Si se saltó alguna fila, recorta las filas sobrantes
	data.X = data.X.rowRange(0, row).clone();
	data.y = data.y.rowRange(0, row).clone();
	return data;
}

cv::Mat select_columns(const cv::Mat &X, int start, int end){
	return X.colRange(start, end).clone();
}