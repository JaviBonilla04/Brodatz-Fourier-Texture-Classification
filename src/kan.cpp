#define _USE_MATH_DEFINES  
#include <cmath>

#include "config.hpp"
#include "data_loader.hpp"
#include "evaluation.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

/*
C:\Users\JB\Desktop\ComputerVision\texture_classification\src>cl kan.cpp ..\src\data_loader.cpp ..\src\evaluation.cpp /EHsc /std:c++20 /MD /I ..\src /I "%OPENCV_DIR%\include" /I C:\libtorch\include /I C:\libtorch\include\torch\csrc\api\include /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" /LIBPATH:C:\libtorch\lib opencv_world4130.lib torch.lib torch_cpu.lib c10.lib
*/


torch::Tensor to_tensor_float(const cv::Mat &X){
	return torch::from_blob(X.data, {X.rows, X.cols}, torch::kFloat).clone();
}

torch::Tensor to_tensor_labels(const cv::Mat &y){
	return torch::from_blob(y.data, {y.rows}, torch::kInt).to(torch::kLong);
}

// tensor de predicciones (N) -> cv::Mat N x 1 CV_32S, para usar evaluation.cpp
cv::Mat to_mat(const torch::Tensor &t){
	torch::Tensor t32 = t.to(torch::kInt).contiguous();
	return cv::Mat((int)t32.size(0), 1, CV_32S, t32.data_ptr<int>()).clone();
}

struct Config{
	std::string name;
	int start, end;
};

const std::vector<std::string> FEATURE_NAMES = {
	"g_entropy","g_concentration","g_mean_radial_freq","g_std_radial_freq",
	"r1_entropy","r1_concentration","r1_mean_radial_freq","r1_std_radial_freq",
	"r2_entropy","r2_concentration","r2_mean_radial_freq","r2_std_radial_freq",
	"r3_entropy","r3_concentration","r3_mean_radial_freq","r3_std_radial_freq"};

struct FourierKANImpl : torch::nn::Module {
	int64_t in_features, out_features, n_freqs;
	torch::Tensor a, b, bias, k;

	FourierKANImpl(int64_t in, int64_t out, int64_t K) : in_features(in), out_features(out), n_freqs(K) {
		double scale = 1.0 / std::sqrt((double)(in * K));
		a    = register_parameter("a", torch::randn({in, out, K}) * scale);
		b    = register_parameter("b", torch::randn({in, out, K}) * scale);
		bias = register_parameter("bias", torch::zeros({out}));
		// frecuencias 1..K (no se entrenan)
		k    = register_buffer("k", torch::arange(1, K + 1, torch::kFloat));
	}

	// x: (lote, in) ya escalado a [-pi/2, pi/2]  ->  (lote, out)
	torch::Tensor forward(const torch::Tensor &x){
		torch::Tensor kx = x.unsqueeze(-1) * k;                        // (lote, in, K)
		return torch::einsum("bik,iok->bo", {torch::cos(kx), a})
		     + torch::einsum("bik,iok->bo", {torch::sin(kx), b})
		     + bias;
	}

	// Penalización de suavidad: sum k^2 (a^2 + b^2). Castiga las frecuencias altas.
	torch::Tensor smoothness() const {
		torch::Tensor k2 = (k * k).view({1, 1, n_freqs});
		return (k2 * (a * a + b * b)).sum();
	}

	// Curva phi_ij evaluada en los puntos xs (1D): devuelve (len(xs),)
	torch::Tensor curve(int64_t i, int64_t j, const torch::Tensor &xs) const {
		torch::Tensor kx = xs.unsqueeze(-1) * k;                       // (n, K)
		return torch::cos(kx).matmul(a.index({i, j})) + torch::sin(kx).matmul(b.index({i, j}));
	}
};
TORCH_MODULE(FourierKAN);

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

	std::ofstream file(SAVE_METRICS+"fourier_kan.csv");
	file << "model,n_features,f1_macro_test,f1_macro_rotated,percent_changed,n_freqs,epochs,lr,lambda\n";
	file << std::setprecision(10);

	std::cout << std::fixed << std::setprecision(4);
	const int n_classes = (int)class_map.size();
	const std::vector<std::string> names = class_names(class_map);

	for(const Config &c : configs) {
		torch::Tensor Xtr = to_tensor_float(select_columns(train.X, c.start, c.end));
		torch::Tensor mean = Xtr.mean(0);
		torch::Tensor sd = Xtr.std(0);

		torch::Tensor Xte = to_tensor_float(select_columns(test.X, c.start, c.end));
		torch::Tensor Xrot = to_tensor_float(select_columns(rot.X, c.start, c.end));

		auto prepare = [&](const torch::Tensor &X){
			torch::Tensor z = (X-mean)/sd;
			return z.clamp(-3,3)*(M_PI/6.0);
		};

		torch::Tensor Xtr_in  = prepare(Xtr);
		torch::Tensor Xte_in = prepare(Xte);
		torch::Tensor Xrot_in = prepare(Xrot);

		torch::Tensor ytr = to_tensor_labels(train.y);

		torch::manual_seed(42);
		FourierKAN model(Xtr_in.size(1), n_classes, 4);
		torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));

		for (int epoch = 0; epoch < 300; epoch++){
			opt.zero_grad();
			torch::Tensor loss = torch::nn::functional::cross_entropy(model->forward(Xtr_in), ytr) + 1e-4 * model->smoothness();
			loss.backward();
			opt.step();
			if (epoch % 50 == 0) std::cout << "epoch " << epoch << " loss " << loss.item<double>() << "\n";
		}

		torch::NoGradGuard no_grad;
		
		torch::Tensor kx  = Xtr_in.unsqueeze(-1) * model->k;                          // (N, in, K)
		torch::Tensor phi = torch::einsum("bik,iok->bio", {torch::cos(kx), model->a})
		                  + torch::einsum("bik,iok->bio", {torch::sin(kx), model->b});   // (N, in, out)
		torch::Tensor imp = phi.abs().mean(0).sum(1);                                    // (in)
		imp = 100.0 * imp / imp.sum();
		std::ofstream fi(SAVE_METRICS + "kan_importance_" + c.name + ".csv");
		fi << "feature,importance_percent\n";
		for (int i = 0; i < imp.size(0); i++)
			fi << FEATURE_NAMES[c.start + i] << "," << imp[i].item<double>() << "\n";
	

		torch::Tensor pred = model->forward(Xte_in).argmax(1);

		// Saving curves for interpretability
		{
			const int n_pts = 100;
			std::ofstream fc(SAVE_METRICS + "kan_curves_" + c.name + ".csv");
			fc << "feature,class,x_original,phi\n";
			for (int i = 0; i < Xtr_in.size(1); i++){
				torch::Tensor col = Xtr_in.select(1, i);
				torch::Tensor xs  = torch::linspace(col.min().item<double>(), col.max().item<double>(), n_pts);
				torch::Tensor x_orig = mean[i] + sd[i] * (xs / (M_PI / 6.0));
				auto xo = x_orig.accessor<float, 1>();
				for (int j = 0; j < n_classes; j++){
					torch::Tensor phi_ij = model->curve(i, j, xs);
					auto ph = phi_ij.accessor<float, 1>();
					for (int p = 0; p < n_pts; p++)
						fc << FEATURE_NAMES[c.start + i] << "," << names[j] << ","
						   << xo[p] << "," << ph[p] << "\n";
				}
			}
		}

		torch::Tensor pred_rot = model->forward(Xrot_in).argmax(1);
		
		
		cv::Mat pred_m = to_mat(pred), pred_rot_m = to_mat(pred_rot);
		cv::Mat cm = confusion_matrix(test.y, pred_m, n_classes);
		cv::Mat cm_rot = confusion_matrix(rot.y,  pred_rot_m, n_classes);
		save_confusion_matrix(cm,     names, SAVE_METRICS + "cm_kan_" + c.name + "_test.csv");
		save_confusion_matrix(cm_rot, names, SAVE_METRICS + "cm_kan_" + c.name + "_rotated.csv");

		double f1 = f1_macro(cm), f1_rot = f1_macro(cm_rot);
		double chg = 100.0 * percent_changed(pred_m, pred_rot_m);

		file << c.name << "," << (c.end - c.start) << "," << f1 << "," << f1_rot << "," << chg << ",4,300,0.01,0.0001\n";
		std::cout << c.name << " | F1 test: " << f1 << " | F1 rotated: " << f1_rot << " | changed: " << chg << " %\n";

	}

}