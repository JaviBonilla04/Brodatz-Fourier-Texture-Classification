#include <torch/torch.h>
#include <iostream>

int main(){
	torch::Tensor t = torch::rand({2, 3});
	std::cout << t << "\n";
}