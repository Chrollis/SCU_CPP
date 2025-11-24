#ifndef FULL_CONNECT_LAYER_HPP
#define FULL_CONNECT_LAYER_HPP

#include "activation_function.hpp"
#include <filesystem>
#include <Eigen/Dense>

namespace chr {
	class full_connect_layer {
	private:
		size_t in_size_; // 输入维度
		size_t out_size_; // 输出维度
		activation_function afunc_; // 激活函数
		Eigen::MatrixXd weights_; // 权重矩阵
		Eigen::VectorXd biases_; // 偏置向量
		Eigen::VectorXd input_; // 输入数据
		Eigen::VectorXd gradient_; // 梯度
		Eigen::VectorXd feature_vector_; // 特征向量(激活后)
		Eigen::VectorXd linear_outcome_; // 线性输出(激活前)
	public:
		full_connect_layer(size_t in_size, size_t out_size, activation_function_type activate_type = activation_function_type::relu) : in_size_(in_size), out_size_(out_size), afunc_(activate_type) { initialize_weights(); }
		Eigen::VectorXd forward(const Eigen::VectorXd& input);
		Eigen::VectorXd backward(const Eigen::VectorXd& gradient, double learning_rate, bool is_output_layer = false, size_t label = 0);
		void weights_update(double learning_rate);
		void save(const std::filesystem::path& path) const;
		void load(const std::filesystem::path& path);
	private:
		void initialize_weights();
	};

	// Example:
	// input: 400 * 1
	// output : 120 * 1
	// weights: 120 * 400
	// bias: 120 * 1
}

#endif // !FULL_CONNECT_LAYER_HPP
