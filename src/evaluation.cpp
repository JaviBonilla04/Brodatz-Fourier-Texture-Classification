#include "evaluation.hpp"

// rows are real classes and cols are the predicted class when attempting to predict that class
cv::Mat confusion_matrix(const cv::Mat &y_true, const cv::Mat &y_pred, int n){
	cv::Mat cm = cv::Mat::zeros(n, n, CV_32S);
	for(int i = 0; i < y_true.rows; i++){
		int real = y_true.at<int>(i, 0), predict = y_pred.at<int>(i, 0);
		cm.at<int>(real, predict) += 1;
	} 
	return cm;
}

std::vector<double> f1_per_class(const cv::Mat &cm){
	std::vector<double> f1_class(cm.rows, 0);
	for(int c = 0; c < cm.rows; c++){
		int fp = 0, fn = 0; 
		for(int r = 0; r < cm.cols; r++){
			fp += cm.at<int>(r,c);
			fn += cm.at<int>(c,r);
		}
		int tp = cm.at<int>(c,c);
		fp -= tp; fn -= tp;
		f1_class[c] = 2.0*tp/(2.0*tp+fp+fn);
	}
	return f1_class;
}

// the average of the f1 per class
double f1_macro(const cv::Mat &cm){
	std::vector<double> f1 = f1_per_class(cm);
	double sum = 0.0;
	for(double v: f1) sum+=v;
	return sum/f1.size();
}

double percent_changed(const cv::Mat &pred, const cv::Mat &pred_rot){
	int diff = 0;
	for(int i = 0; i < pred.rows; i++){
		if(pred.at<int>(i) != pred_rot.at<int>(i)) diff++;
	}
	return static_cast<double>(diff)/pred.rows;
}