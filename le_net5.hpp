#ifndef LE_NET5_HPP
#define LE_NET5_HPP

#include "mnist_data.hpp"
#include "convolve_layer.hpp"
#include "pool_layer.hpp"
#include "full_connect_layer.hpp"

namespace chr {
	class le_net5 {
	private:
		// 网络层定义(LeNet-5结构)
		convolve_layer conv1_; // 第一卷积层
		pool_layer pool1_; // 第一池化层
		convolve_layer conv2_; // 第二卷积层  
		pool_layer pool2_; // 第二池化层
		full_connect_layer fc1_; // 第一全连接层
		full_connect_layer fc2_; // 第二全连接层
		full_connect_layer fc3_; // 第三全连接层(输出层)
	public:
		le_net5();
		Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input);
		std::vector<Eigen::MatrixXd> backward(size_t label, double learning_rate);
		double train(const std::vector<mnist_data>& dataset, size_t epochs, double learning_rate);
		size_t predict(const Eigen::VectorXd& output); // 预测函数
		void save(const std::filesystem::path& path);
		void load(const std::filesystem::path& path);
	private:
		Eigen::VectorXd flatten(const std::vector<Eigen::MatrixXd>& matrixs); // 展平操作(多维转一维)
		std::vector<Eigen::MatrixXd> counterflatten(const Eigen::VectorXd& vector, size_t channels, size_t rows, size_t cols); // 反展平操作(一维转多维)
		double cross_entropy_loss(const Eigen::VectorXd& output, size_t label); // 交叉熵损失函数
	};
}

#endif // !LE_NET5_HPP
