#include "full_connect_layer.hpp"
#include <random>
#include <fstream>

namespace chr {
	Eigen::VectorXd full_connect_layer::forward(const Eigen::VectorXd& input) {
		input_ = input;
		// 线性计算：output = weights * input + biases
		linear_outcome_ = weights_ * input_ + biases_;
		feature_vector_ = linear_outcome_.unaryExpr([this](double a) { return afunc_(a); });
		return feature_vector_;
	}
	Eigen::VectorXd full_connect_layer::backward(const Eigen::VectorXd& gradient, double learning_rate, bool is_output_layer, size_t label) {
		if (is_output_layer) {
			// 如果是输出层，计算交叉熵损失梯度
			Eigen::VectorXd target = Eigen::VectorXd::Zero(out_size_);
			if (label >= 0 && label < out_size_) {
				target(label) = 1.0; // 创建one-hot标签
			}
			// 输出层梯度 = 预测值 - 真实值
			gradient_ = feature_vector_ - target;
		}
		else {
			// 隐藏层梯度 = 上层梯度 × 激活函数导数
			gradient_ = gradient.cwiseProduct(linear_outcome_.unaryExpr([this](double a) { return afunc_[a]; }));
		}
		Eigen::VectorXd next_gradient = weights_.transpose() * gradient_;
		weights_update(learning_rate);
		return next_gradient;
	}
	void full_connect_layer::weights_update(double learning_rate) {
		// 权重更新：weights -= 学习率 * 梯度 * 输入转置
		weights_ -= learning_rate * gradient_ * input_.transpose();
		// 偏置更新：biases -= 学习率 * 梯度
		biases_ -= learning_rate * gradient_;
	}
	void full_connect_layer::save(const std::filesystem::path& path) const {
		std::ofstream file(path);
		if (file.is_open()) {
			file << in_size_ << " " << out_size_ << "\n";
			file << weights_ << "\n";
			file << biases_.transpose() << "\n";
			file.close();
		}
	}
	void full_connect_layer::load(const std::filesystem::path& path) {
		std::ifstream file(path);
		if (file.is_open()) {
			size_t in_size, out_size;
			file >> in_size >> out_size;
			weights_.resize(out_size, in_size);
			biases_.resize(out_size);
			for (size_t i = 0; i < out_size; ++i) {
				for (size_t j = 0; j < in_size; ++j) {
					file >> weights_(i, j);
				}
			}
			for (size_t i = 0; i < out_size; ++i) {
				file >> biases_(i);
			}
			file.close();
		}
	}
	void full_connect_layer::initialize_weights() {
		weights_.resize(out_size_, in_size_);
		biases_.resize(out_size_);
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::normal_distribution<double> distribution(0.0, 0.01); // 使用小方差高斯分布
		for (size_t i = 0; i < out_size_; ++i) {
			for (size_t j = 0; j < in_size_; ++j) {
				weights_(i, j) = distribution(generator);
			}
			biases_(i) = distribution(generator);
		}
	}
}