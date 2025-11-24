#include "convolve_layer.hpp"

namespace chr {
	convolve_layer::convolve_layer(size_t in_channels, size_t kernel_size, size_t out_channels, size_t padding, size_t stride, activation_function_type activate_type)
		: in_channels_(in_channels), kernel_size_(kernel_size), out_channels_(out_channels), padding_(padding), stride_(stride), afunc_(activate_type) {
		for (size_t i = 0; i < out_channels; ++i) {
			filters_.emplace_back(in_channels, kernel_size);
			// 使用He初始化方法
			filters_.back().initialize_He(in_channels * kernel_size * kernel_size);
		}
	}
	std::vector<Eigen::MatrixXd> convolve_layer::forward(const std::vector<Eigen::MatrixXd>& input) {
		input_ = padding(input, padding_);
		convolve_outcome_.clear();
		feature_map_.clear();
		for (const auto& filter : filters_) {
			// 计算输出特征图尺寸
			size_t rows = (input_[0].rows() - kernel_size_) / stride_ + 1;
			size_t cols = (input_[0].cols() - kernel_size_) / stride_ + 1;
			Eigen::MatrixXd channel_result = Eigen::MatrixXd::Zero(rows, cols);
			// 多通道卷积：每个输入通道与对应的卷积核卷积，然后求和
			for (size_t i = 0; i < in_channels_; ++i) {
				Eigen::MatrixXd conv_result = convolve(input_[i], filter.kernels[i], stride_);
				channel_result += conv_result;
			}
			channel_result = channel_result.unaryExpr([filter](double a) { return a + filter.bias; });
			convolve_outcome_.push_back(channel_result);
			feature_map_.push_back(channel_result.unaryExpr([this](double x) { return afunc_(x); }));
		}
		return feature_map_;
	}
	std::vector<Eigen::MatrixXd> convolve_layer::backward(const std::vector<Eigen::MatrixXd>& gradient, double learning_rate, bool is_last_conv) {
		std::vector<Eigen::MatrixXd> propagated_gradient;
		if (is_last_conv) {
			// 如果是最后一层卷积层，直接使用传入的梯度
			propagated_gradient = gradient;
		}
		else {
			// 否则应用激活函数的导数
			propagated_gradient = apply_activation_derivative(gradient);
		}
		std::vector<Eigen::MatrixXd> next_gradient;
		for (size_t i = 0; i < in_channels_; i++) {
			Eigen::MatrixXd each_channel = Eigen::MatrixXd::Zero(input_[0].rows(), input_[0].cols());
			for (size_t j = 0; j < out_channels_; j++) {
				// 旋转卷积核180度(用于反向传播)
				Eigen::MatrixXd rotated_kernel = filters_[j].kernels[i].reverse();
				size_t pad_size = kernel_size_ - 1;
				Eigen::MatrixXd padded = padding({ gradient[j] }, pad_size)[0];
				each_channel += convolve(padded, rotated_kernel, 1);
			}
			next_gradient.push_back(std::move(each_channel));
		}
		weights_update(learning_rate, propagated_gradient);
		// 移除填充后返回给前一层
		return remove_padding(next_gradient, padding_);
	}
	void convolve_layer::weights_update(double learning_rate, const std::vector<Eigen::MatrixXd>& gradient) {
		for (size_t i = 0; i < out_channels_;i++) {
			for (size_t j = 0; j < in_channels_;j++) {
				// 更新卷积核权重：权重 -= 学习率 * 输入与梯度的卷积
				filters_[i].kernels[j] -= learning_rate * convolve(input_[j], gradient[i], 1);
			}
			// 更新偏置：偏置 -= 学习率 * 梯度总和
			filters_[i].bias -= learning_rate * gradient[i].sum();
		}
	}
	void convolve_layer::save(const std::filesystem::path& path) const {
		std::filesystem::create_directory(path);
		for (size_t i = 0; i < filters_.size(); ++i) {
			filters_[i].save(path / ("filter_" + std::to_string(i) + ".txt"));
		}
	}
	void convolve_layer::load(const std::filesystem::path& path) {
		std::filesystem::create_directory(path);
		for (size_t i = 0; i < filters_.size(); ++i) {
			filters_[i].load(path / ("filter_" + std::to_string(i) + ".txt"));
		}
	}
	std::vector<Eigen::MatrixXd> convolve_layer::padding(const std::vector<Eigen::MatrixXd>& input, size_t circle_num, double fill_num) const {
		std::vector<Eigen::MatrixXd> result;
		for (const auto& channel : input) {
			size_t new_rows = channel.rows() + 2 * circle_num;
			size_t new_cols = channel.cols() + 2 * circle_num;
			Eigen::MatrixXd padded = Eigen::MatrixXd::Constant(new_rows, new_cols, fill_num);
			// 将原始数据放入中心位置
			padded.block(circle_num, circle_num, channel.rows(), channel.cols()) = channel;
			result.push_back(padded);
		}
		return result;
	}
	Eigen::MatrixXd convolve_layer::convolve(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel, size_t stride) const {
		if (input.rows() < kernel.rows() || input.cols() < kernel.cols()) {
			throw std::invalid_argument("input dimension is smaller that kernel dimension");
		}
		long output_rows = static_cast<long>((input.rows() - kernel.rows()) / stride) + 1;
		long output_cols = static_cast<long>((input.cols() - kernel.cols()) / stride) + 1;
		if (output_rows <= 0 || output_cols <= 0) {
			throw std::invalid_argument("outcome dimension is not positive");
		}
		Eigen::MatrixXd result = Eigen::MatrixXd::Zero(output_rows, output_cols);
		Eigen::Map<const Eigen::VectorXd> kernel_flat(kernel.data(), kernel.size());
		for (long i = 0; i < output_rows; i++) {
			for (long j = 0; j < output_cols; j++) {
				long start_row = i * stride;
				long start_col = j * stride;
				if (start_row + kernel.rows() <= input.rows() &&
					start_col + kernel.cols() <= input.cols()) {
					Eigen::MatrixXd block = input.block(start_row, start_col, kernel.rows(), kernel.cols());
					Eigen::Map<const Eigen::VectorXd> block_flat(block.data(), block.size());
					result(i, j) = block_flat.dot(kernel_flat);
				}
			}
		}
		return result;
	}
	std::vector<Eigen::MatrixXd> convolve_layer::apply_activation_derivative(const std::vector<Eigen::MatrixXd>& gradient) {
		std::vector<Eigen::MatrixXd> result;
		for (size_t i = 0; i < gradient.size(); ++i) {
			result.push_back(gradient[i].cwiseProduct(convolve_outcome_[i].unaryExpr([this](double x) { return afunc_[x]; })));
		}
		return result;
	}
	std::vector<Eigen::MatrixXd> convolve_layer::remove_padding(const std::vector<Eigen::MatrixXd>& input, size_t padding) {
		if (padding == 0) return input;
		std::vector<Eigen::MatrixXd> result;
		for (const auto& matrix : input) {
			result.push_back(matrix.block(padding, padding, matrix.rows() - 2 * padding, matrix.cols() - 2 * padding));
		}
		return result;
	}
}