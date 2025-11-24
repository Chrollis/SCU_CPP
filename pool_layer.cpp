#include "pool_layer.hpp"

namespace chr {
	std::vector<Eigen::MatrixXd> pool_layer::forward(const std::vector<Eigen::MatrixXd>& input) {
		input_ = input;
		feature_map_.clear();
		record_.clear();
		size_t rows = (input_[0].rows() - core_size_) / stride_ + 1;
		size_t cols = (input_[0].cols() - core_size_) / stride_ + 1;
		for (const auto& channel : input_) {
			Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows, cols);
			if (type_ == pooling_type::max) {
				// 最大池化：需要记录最大值位置
				record_.push_back(Eigen::MatrixXd::Zero(channel.rows(), channel.cols()));
				max_pooling(channel, output, record_.back());
			}
			else {
				average_pooling(channel, output);
			}
			feature_map_.push_back(output);
		}
		return feature_map_;
	}
	std::vector<Eigen::MatrixXd> pool_layer::backward(const std::vector<Eigen::MatrixXd>& gradient) {
		std::vector<Eigen::MatrixXd> next_gradient;
		for (size_t i = 0; i < gradient.size(); i++) {
			if (type_ == pooling_type::max) {
				next_gradient.push_back(max_backward(gradient[i], record_[i]));
			}
			else {
				next_gradient.push_back(average_backward(gradient[i]));
			}
		}
		return next_gradient;
	}
	void pool_layer::max_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output, Eigen::MatrixXd& record) const {
		for (size_t i = 0; i < output.rows(); ++i) {
			for (size_t j = 0; j < output.cols(); ++j) {
				double max = std::numeric_limits<double>::lowest();
				size_t row = 0, col = 0;
				// 在池化窗口内寻找最大值
				for (size_t m = 0; m < core_size_; ++m) {
					for (size_t n = 0; n < core_size_; ++n) {
						double val = input(i * stride_ + m, j * stride_ + n);
						if (val > max) {
							max = val;
							row = i * stride_ + m;
							col = j * stride_ + n;
						}
					}
				}
				output(i, j) = max;
				record(row, col) = 1.0; // 在记录矩阵中标记最大值位置
			}
		}
	}
	void pool_layer::average_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output) const {
		double divisor = core_size_ * core_size_;
		for (size_t i = 0; i < output.rows(); ++i) {
			for (size_t j = 0; j < output.cols(); ++j) {
				double sum = 0.0;
				// 计算池化窗口内所有元素的平均值
				for (size_t m = 0; m < core_size_; ++m) {
					for (size_t n = 0; n < core_size_; ++n) {
						sum += input(i * stride_ + m, j * stride_ + n);
					}
				}
				output(i, j) = sum / divisor;
			}
		}
	}
	Eigen::MatrixXd pool_layer::max_backward(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& record) const {
		Eigen::MatrixXd next_gradient = Eigen::MatrixXd::Zero(input_[0].rows(), input_[0].cols());
		for (size_t i = 0; i < gradient.rows(); ++i) {
			for (size_t j = 0; j < gradient.cols(); ++j) {
				for (size_t m = 0; m < core_size_; ++m) {
					for (size_t n = 0; n < core_size_; ++n) {
						size_t row = i * stride_ + m;
						size_t col = j * stride_ + n;
						// 梯度只传递给前向传播时最大值的位置
						next_gradient(row, col) += gradient(i, j) * record(row, col);
					}
				}
			}
		}
		return next_gradient;
	}
	Eigen::MatrixXd pool_layer::average_backward(const Eigen::MatrixXd& gradient) {
		Eigen::MatrixXd next_gradient = Eigen::MatrixXd::Zero(input_[0].rows(), input_[0].cols());
		double divisor = core_size_ * core_size_;
		for (size_t i = 0; i < gradient.rows(); ++i) {
			for (size_t j = 0; j < gradient.cols(); ++j) {
				double avg = gradient(i, j) / divisor;
				for (size_t m = 0; m < core_size_; ++m) {
					for (size_t n = 0; n < core_size_; ++n) {
						// 将梯度平均分配到池化窗口内的所有位置
						next_gradient(i * stride_ + m, j * stride_ + n) += avg;
					}
				}
			}
		}
		return next_gradient;
	}
}