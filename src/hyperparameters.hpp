#ifndef HYPERPARAMETERS_HPP
#define HYPERPARAMETERS_HPP

#include <string>

//-- Decision Tree
inline constexpr int MAX_DEPTH_dt = 20;
inline constexpr int MIN_SAMPLE_CNT_dt = 5;
inline constexpr int CV_FOLDS_dt = 0; //deactivate internal trimming

inline constexpr bool USE_SURROGATES_dt = false;



#endif