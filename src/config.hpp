#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

inline constexpr int SIDE = 64;
inline constexpr int N_REGIONS = 3;
inline constexpr int N_FEATURES = 4;

inline const std::string MASKS = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\results\\figures\\masks\\";

inline const std::string DATA = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\data\\";

inline const std::string TRAIN_DATA = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\data\\train_dataset.csv";
inline const std::string TEST_DATA = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\data\\test_dataset.csv";


inline const std::string TRAIN_DATA_DESC = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\data\\train_descriptors_data.csv";
inline const std::string TEST_DATA_DESC = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\data\\test_descriptors_data.csv";

inline const std::string TEST_ROTATED_DATA_DESC = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\data\\test_rotated_desc_data.csv";

inline const std::string SAVE_RESULTS = "C:\\Users\\JB\\Desktop\\ComputerVision\\texture_classification\\results\\";	
inline const std::string SAVE_MODELS = SAVE_RESULTS + "models\\";
inline const std::string SAVE_METRICS = SAVE_RESULTS + "metrics\\";

#endif 