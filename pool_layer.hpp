#ifndef POOL_LAYER_HPP
#define POOL_LAYER_HPP

#include <Eigen/Dense>

namespace chr {
	enum class pooling_type {
		max, // 最大池化
		average // 平均池化
	};

	class pool_layer {
	private:
		size_t core_size_; // 池化核尺寸
		size_t stride_; // 步长
		pooling_type type_; // 池化类型
		std::vector<Eigen::MatrixXd> input_; // 输入数据
		std::vector<Eigen::MatrixXd> feature_map_; // 池化后的特征图
		std::vector<Eigen::MatrixXd> record_; // 记录最大池化的位置(用于反向传播)
	public:
		pool_layer(size_t core_size, size_t stride, pooling_type type = pooling_type::max) : core_size_(core_size), stride_(stride), type_(type) {}
		std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input);
		std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& gradient);
	private:
		void max_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output, Eigen::MatrixXd& record) const;
		void average_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output) const;
		Eigen::MatrixXd max_backward(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& record) const;
		Eigen::MatrixXd average_backward(const Eigen::MatrixXd& gradient);
	};

	// Example:
	// input: 16 * 10 * 10 (the same size of the record)
	// settings: 
	// > core size: 2
	// > stride: 2
	// output: 16 * 5 * 5 (the same size of the feature map)
}

#endif // !POOL_LAYER_HPP
