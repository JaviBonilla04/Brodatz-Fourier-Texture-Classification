#include <filesystem>  // std::filesystem::path, create_directories
#include <fstream>     // std::ofstream
#include <iostream>    // cout, cerr
#include <string>      
#include <vector>     

#include <opencv2/opencv.hpp>

using namespace std;

/*
cl brodatz_crops.cpp /EHsc /std:c++17 /I "%OPENCV_DIR%\include" /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4130.lib

*/

// -- System data
const string FOLDER = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\data\\";

const string TRAIN_FOLDER = "train\\";
const string TEST_FOLDER = "test\\";

const string TRAIN_DATASET = "train_dataset.csv";
const string TEST_DATASET = "test_dataset.csv";

// -- Hyperparameters
const int CROP_SIZE = 64;  	
const float TRAIN_SIZE = 0.7f;

struct Sample{
	string name;
	cv::Mat img;
};

struct Crop{
	string name;
	int row;
	int col;
	cv::Mat img;
};

vector<Sample> load_dataset(const string& dataset_route){
	vector<cv::String> routes;
	cv::glob(dataset_route + "/*.png", routes, false);

	std::vector<Sample> dataset;
	
	for(const auto& r : routes) {
		cv::Mat img = cv::imread(r, cv::IMREAD_GRAYSCALE);
		if(img.empty()){
			std::cerr << "Not able to read image: " << r << "\n";
			continue;
		}
		string name = std::filesystem::path(r).stem().string();
		dataset.push_back({name, img});
	}
	return dataset;
}

void crop_sample(const Sample &s, vector<Crop> &crops, int col_start, int col_end){
	cv::Mat img = s.img;
	for(int i = 0; i + CROP_SIZE <= img.rows; i+=CROP_SIZE){
		for(int j = col_start; j + CROP_SIZE <= col_end; j+=CROP_SIZE){
			cv::Mat curr_crop = img(cv::Rect(j, i, CROP_SIZE, CROP_SIZE));
			crops.push_back({s.name, i, j, curr_crop});
		}
	}
}

void save_metadata(const string &set_class, const Crop &c, std::ofstream &file, const string &folder_route){
	string id_crop = c.name + "_" + to_string(c.row) + "_" + to_string(c.col);
	file << id_crop << "," << c.name << "," << c.row << "," << c.col << "," << set_class << ","<< FOLDER + folder_route + id_crop + ".png"<< "\n";
}	

void save_crop(const Crop &c, const string &folder_route){
	string route = FOLDER + folder_route + c.name + "_" + to_string(c.row) + "_" + to_string(c.col) + ".png";
	if(!cv::imwrite(route, c.img)) {
		std::cerr << "Could not save: " << route << "\n";
	}
}

void create_crops(const vector<Sample>& samples) {
	std::filesystem::remove_all(FOLDER + TRAIN_FOLDER);
	std::filesystem::remove_all(FOLDER + TEST_FOLDER);

	std::filesystem::create_directories(FOLDER + TRAIN_FOLDER);
	std::filesystem::create_directories(FOLDER + TEST_FOLDER);

	std::ofstream file_train(FOLDER+TRAIN_DATASET);
	file_train << "id_crop,name_class,row,col,set,route\n";

	std::ofstream file_test(FOLDER+TEST_DATASET);
	file_test << "id_crop,name_class,row,col,set,route\n";

	for(Sample s : samples){
		int split = (int)(s.img.cols * TRAIN_SIZE); 

		vector<Crop> train_crops, test_crops;
		crop_sample(s, train_crops, 0, split);
		crop_sample(s, test_crops, split, s.img.cols); 
		
		for(const Crop& c : train_crops){
			save_metadata("train", c, file_train, TRAIN_FOLDER);
			save_crop(c, TRAIN_FOLDER);
		}

		for(const Crop& c : test_crops){
			save_metadata("test", c, file_test, TEST_FOLDER);
			save_crop(c, TEST_FOLDER);
		}

		cout << s.name << ": " << train_crops.size() << " train, " << test_crops.size() << " test\n";
	}

}

int main(){
	string dataset_route = FOLDER + "brodatz_png"; 

	vector<Sample> dataset;
	dataset = load_dataset(dataset_route);
	
 	if (dataset.empty()) {
  		cerr << "No images found in " << dataset_route << "\n";
  		return 1;
  	}

	create_crops(dataset);
	
}