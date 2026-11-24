#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <vector>
#include <opencv2/opencv.hpp>


// rows are real classes and cols are the predicted class when attempting to predict that class
cv::Mat confusion_matrix(const cv::Mat &y_true, const cv::Mat &y_pred, int n);

std::vector<double> f1_per_class(const cv::Mat &cm);

double f1_macro(const cv::Mat &cm);

double percent_changed(const cv::Mat &pred, const cv::Mat &pred_rot);

#endif