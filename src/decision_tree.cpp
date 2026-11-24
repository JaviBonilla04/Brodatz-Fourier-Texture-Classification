#include "config.hpp"
#include "hyperparameters.hpp"

#include "data_loader.hpp"
#include "evaluation.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <charconv>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp> 

/*
cl decision_tree.cpp ..\src\data_loader.cpp ..\src\evaluation.cpp /EHsc /std:c++17 /I ..\src /I "%OPENCV_DIR%\include" /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4130.lib
*/

struct Config{
	std::string name;
	int start, end;
};

cv::Ptr<cv::ml::DTrees> create_decisionTree(int max_depth, int min_sample_cnt, int cv_folds, bool use_surrogates){
    cv::Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();
    tree->setMaxDepth(max_depth);
    tree->setMinSampleCount(min_sample_cnt);
    tree->setCVFolds(cv_folds);
    tree->setUseSurrogates(use_surrogates);
    return tree;
}

cv::Mat predict(const cv::Ptr<cv::ml::DTrees> &tree, const cv::Mat &X){
	cv::Mat pred_f, pred;
	tree->predict(X, pred_f);
	pred_f.convertTo(pred, CV_32S);
	return pred;
}

void save_eval(std::ofstream &file, std::string tree_name, int n_features, double f1_macro_pred, double f1_macro_pred_rot, double per_chg, int max_depth, int min_sample_cnt){
	file << tree_name <<","<< n_features <<","<< f1_macro_pred <<","<< f1_macro_pred_rot <<","<< per_chg <<","<< max_depth <<","<< min_sample_cnt;
	file << "\n";

}

void save_confusion_matrix(const cv::Mat &cm, const std::vector<std::string> &names, const std::string &path){
	std::ofstream f(path);
	f << "true\\pred";
	for (const auto &n : names) f << "," << n;
	f << "\n";
	for (int i = 0; i < cm.rows; i++){
		f << names[i];
		for (int j = 0; j < cm.cols; j++) f << "," << cm.at<int>(i, j);
		f << "\n";
	}
}

int main(){
	auto class_map = build_class_map(TRAIN_DATA_DESC);
	FeatureSet train = load_features(TRAIN_DATA_DESC, class_map);
	FeatureSet test = load_features(TEST_DATA_DESC, class_map);
	FeatureSet rot = load_features(TEST_ROTATED_DATA_DESC, class_map);

	std::vector<Config> configs = {
		{"global", 0, 4},
		{"regional", 4, 16},
		{"combined", 0, 16}
	};

	std::filesystem::create_directories(SAVE_METRICS);
	std::filesystem::create_directories(SAVE_MODELS);

	std::ofstream file(SAVE_METRICS+"decision_tree.csv");
	file << "tree,n_features,f1_macro_test,f1_macro_rotated,percent_changed,max_depth,min_sample_count\n";
	file << std::setprecision(10);

	std::cout << std::fixed << std::setprecision(4);
	const int n_classes = (int)class_map.size();
	const std::vector<std::string> names = class_names(class_map);

	for(const Config &c : configs){
		auto tree = create_decisionTree(MAX_DEPTH_dt, MIN_SAMPLE_CNT_dt, CV_FOLDS_dt, USE_SURROGATES_dt);
		tree->train(select_columns(train.X, c.start, c.end), cv::ml::ROW_SAMPLE, train.y);

		cv::Mat pred = predict(tree, select_columns(test.X, c.start, c.end));
		cv::Mat pred_rot = predict(tree, select_columns(rot.X, c.start, c.end));

		cv::Mat cm_pred = confusion_matrix(test.y, pred, n_classes);
		cv::Mat cm_pred_rot = confusion_matrix(rot.y, pred_rot, n_classes);
		save_confusion_matrix(cm_pred, names, SAVE_METRICS + "cm_" + c.name + "_test.csv");
		save_confusion_matrix(cm_pred_rot, names, SAVE_METRICS + "cm_" + c.name + "_rotated.csv");

		double f1_macro_pred = f1_macro(cm_pred);
		double f1_macro_pred_rot = f1_macro(cm_pred_rot);

		double per_chg = percent_changed(pred, pred_rot);

		save_eval(file, c.name, c.end - c.start, f1_macro_pred, f1_macro_pred_rot, 100.0 * per_chg, MAX_DEPTH_dt, MIN_SAMPLE_CNT_dt);

		std::cout << c.name
		          << " | F1 test: " << f1_macro_pred
		          << " | F1 rotated: " << f1_macro_pred_rot
		          << " | changed: " << 100.0 * per_chg << " %\n";

		tree->save(SAVE_MODELS + "decision_tree_" + c.name + ".yml");
	}
}