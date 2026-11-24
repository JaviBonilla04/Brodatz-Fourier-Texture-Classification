#include "config.hpp"
#include "masks.hpp"
#include "features.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <string_view>
#include <charconv>
#include <iomanip>
#include <string>

#include <opencv2/opencv.hpp>

struct CropMetadata{
	std::string name_class;
	int row;
	int col;
	std::string set;
	std::string route;
};

struct Crop{
	std::string id_crop;
	CropMetadata metadata;
	cv::Mat img;
	std::vector<double> descriptor = {};
};

std::vector<std::string_view> divide_line(std::string_view line, char delimitator=','){
	std::vector<std::string_view> cols;
	size_t first = 0, end = line.find(',');

	while(end != std::string_view::npos){
		cols.emplace_back(line.substr(first, end-first));
		first = end+1;
		end = line.find(delimitator, first);
	}
	cols.emplace_back(line.substr(first));
	return cols;
}

template <typename T>
T toNum(std::string_view sv){
	T value{};
	std::from_chars(sv.data(), sv.data()+sv.size(), value);
	return value;
}

cv::Mat load_image(const std::string &img_route){
	cv::Mat img = cv::imread(img_route, cv::IMREAD_GRAYSCALE);
	if(img.empty()) {
		std::cerr << "Image can not be loaded" << "\n";
		return cv::Mat();
	}
	return img;	
}

std::vector<Crop> load_dataset(const std::string route){
	std::vector<Crop> dataset;
	std::ifstream arch(route);
	if(!arch.is_open()) {
		std::cerr <<  "There is no dataset" << "\n";
		return dataset;
	}

	std::string line;
	std::getline(arch, line);

	while(std::getline(arch, line)){
		if(!line.empty() && line.back() == '\r') line.pop_back();
		if(line.empty()){
			std::cerr << "There is an empty line in the dataset" << "\n";
			continue;
		}
		auto cols = divide_line(line);
		if(cols.size() >= 6){
			cv::Mat img = load_image(std::string(cols[5]));
			if(img.empty()) continue;
			// id_crop,name_class,row,col,set,route
			dataset.emplace_back(
				Crop{
					std::string(cols[0]), //id_crop
					CropMetadata{
						std::string(cols[1]), //name_class
						toNum<int>(cols[2]), //row
						toNum<int>(cols[3]), //col
						std::string(cols[4]), //set
						std::string(cols[5]) //route
					},
					img
				}
			);
		}
	}
	return dataset;
}

std::vector<cv::Mat> run_masks(int side, int n_regions){
	std::pair<double,double> center = mask_center(side);
	std::vector<double> rads = calc_radius(side, n_regions);
	std::vector<cv::Mat> masks = create_masks(center, rads, side);
	return masks;
}

void save_crop_descriptor(const Crop &crop, std::ofstream &file){
	file << crop.id_crop << "," << crop.metadata.name_class << ","
	     << crop.metadata.row << "," << crop.metadata.col << ","
	     << crop.metadata.set << "," << crop.metadata.route;
	for (double v : crop.descriptor) file << "," << v;
	file << "\n";
}

void run(const std::string &data_route, int side, int n_regions, const std::string &save_route, bool rotate=false){
	std::vector<cv::Mat> masks = run_masks(side, n_regions);
	std::vector<Crop> dataset = load_dataset(data_route);
	
	std::pair<double,double> spec_center = spectrum_center(side, side);

	std::string desc_global = "g_entropy,g_concentration,g_mean_radial_freq,g_std_radial_freq";
	std::string desc_r1 = "r1_entropy,r1_concentration,r1_mean_radial_freq,r1_std_radial_freq";
	std::string desc_r2 = "r2_entropy,r2_concentration,r2_mean_radial_freq,r2_std_radial_freq";
	std::string desc_r3 = "r3_entropy,r3_concentration,r3_mean_radial_freq,r3_std_radial_freq";

	std::ofstream file(save_route);
	file << "id_crop,name_class,row,col,set,route," + desc_global + "," + desc_r1 + "," + desc_r2 + "," + desc_r3 + "\n";
	file << std::setprecision(10);

	for(Crop &crop : dataset) {
		cv::Mat img;
		if(rotate) cv::rotate(crop.img, img, cv::ROTATE_90_CLOCKWISE);
		else img = crop.img;
		crop.descriptor = extract_all(img, masks, spec_center);
		save_crop_descriptor(crop, file);
	}
}


int main(){
	run(TRAIN_DATA, SIDE, N_REGIONS, TRAIN_DATA_DESC, false);
	std::cout << "train descriptor created, saved on: " + TRAIN_DATA_DESC + "\n";

	run(TEST_DATA, SIDE, N_REGIONS, TEST_DATA_DESC, false);
	std::cout << "test descriptor created, saved on: " + TEST_DATA_DESC + "\n";

	run(TEST_DATA, SIDE, N_REGIONS, TEST_ROTATED_DATA_DESC, true);
	std::cout << "test rotated descriptor created, saved on: " + TEST_ROTATED_DATA_DESC + "\n";

}