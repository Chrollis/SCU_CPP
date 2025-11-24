#include "le_net5.hpp"

namespace chr {
	le_net5::le_net5() 
		: conv1_(1, 5, 6, 2, 1, activation_function_type::lrelu), // 输入通道1，输出通道6，5x5卷积核
		pool1_(2, 2), // 2x2最大池化，步长2
		conv2_(6, 5, 16, 0, 1, activation_function_type::lrelu), // 输入通道6，输出通道16，5x5卷积核
		pool2_(2, 2), // 2x2最大池化，步长2
		fc1_(400, 120, activation_function_type::lrelu), // 2x2最大池化，步长2
		fc2_(120, 84, activation_function_type::lrelu), // 120->84全连接  
		fc3_(84, 10, activation_function_type::lrelu) { // 84->10全连接(输出层)
	}
	Eigen::VectorXd le_net5::forward(const std::vector<Eigen::MatrixXd>& input) {
		auto a1 = conv1_.forward(input);
		auto p1 = pool1_.forward(a1);
		auto a2 = conv2_.forward(p1);
		auto p2 = pool2_.forward(a2);
		Eigen::VectorXd f = flatten(p2); // 展平为向量
		auto a3 = fc1_.forward(f);
		auto a4 = fc2_.forward(a3);
		return fc3_.forward(a4);
	}
	std::vector<Eigen::MatrixXd> le_net5::backward(size_t label, double learning_rate) {
		auto da4 = fc3_.backward({}, learning_rate, 1, label);
		auto da3 = fc2_.backward(da4, learning_rate);
		auto df = fc1_.backward(da3, learning_rate);
		auto dp2 = counterflatten(df, 16, 5, 5); // 展平为向量
		auto da2 = pool2_.backward(dp2);
		auto dp1 = conv2_.backward(da2, learning_rate, 1);
		auto da1 = pool1_.backward(dp1);
		return conv1_.backward(da1, learning_rate);
	}
	double le_net5::train(const std::vector<mnist_data>& dataset, size_t epochs, double learning_rate) {
		auto total_start_time = std::chrono::high_resolution_clock::now();
		double sum_accuracy = 0.0;
		// 显示当前数据集信息
		std::cout << "Current Dataset：\n";
		size_t k = 0, l = static_cast<size_t>(sqrt(dataset.size()));
		for (const auto& data : dataset) {
			std::cout << std::to_string(data.label()) << ' ';
			if (++k >= l) {
				std::cout << "\n";
				k = 0;
			}
		}
		if (k > 0) {
			std::cout << "\n";
		}
		// 训练循环
		for (size_t epoch = 0; epoch < epochs; ++epoch) {
			auto epoch_start_time = std::chrono::high_resolution_clock::now();
			double loss = 0.0;
			size_t correct = 0;
			std::cout << "Epoch: " << epoch + 1 << "/" << epochs << std::endl;
			size_t total_samples = dataset.size();
			size_t running_count = 0;
			auto last_progress_time = std::chrono::high_resolution_clock::now();
			for (size_t i = 0; i < dataset.size(); ++i) {
				Eigen::VectorXd output = forward({ dataset[i].image() });
				size_t predicted = predict(output);
				double sample_loss = cross_entropy_loss(output, dataset[i].label());
				if (predicted == dataset[i].label()) {
					correct++;
				}
				loss += sample_loss;
				running_count++;
				backward(dataset[i].label(), learning_rate);
				// 进度显示
				if ((i + 1) % 10 == 0 || (i + 1) == total_samples) {
					auto current_time = std::chrono::high_resolution_clock::now();
					auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - epoch_start_time).count() / 1000.0;
					double progress = static_cast<double>(i + 1) / total_samples * 100.0;
					double current_loss = loss / running_count;
					double current_accuracy = static_cast<double>(correct) / running_count * 100.0;
					double estimated_remaining = 0.0;
					if (progress > 0) {
						estimated_remaining = elapsed_time / progress * (100 - progress);
					}
					// 显示进度条
					std::cout << "\r[";
					int bar_width = 30;
					int pos = static_cast<int>(bar_width * progress / 100.0);
					for (int j = 0; j < bar_width; ++j) {
						if (j < pos) std::cout << "=";
						else std::cout << " ";
					}
					std::cout << "] " << std::setw(3) << static_cast<int>(progress) << "%"
						<< " - Loss: " << std::fixed << std::setprecision(4) << current_loss
						<< " - Accuracy: " << std::fixed << std::setprecision(1) << current_accuracy << "%"
						<< " (" << correct << "/" << running_count << ")"
						<< " - Used: " << std::fixed << std::setprecision(1) << elapsed_time << " sec"
						<< " - Rest: " << std::fixed << std::setprecision(1) << estimated_remaining << " sec\t" << std::flush;
					last_progress_time = current_time;
				}
			}
			// 计算epoch统计信息
			auto epoch_end_time = std::chrono::high_resolution_clock::now();
			auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count() / 1000.0;
			double avg_loss = loss / dataset.size();
			double accuracy = static_cast<double>(correct) / dataset.size() * 100.0;
			auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - total_start_time).count() / 1000.0;
			double avg_epoch_time = total_elapsed / (epoch + 1);
			double estimated_total_time = avg_epoch_time * epochs;
			double remaining_time = estimated_total_time - total_elapsed;
			// 输出epoch结果
			std::cout << std::endl << "Epoch: " << epoch + 1 << " finished，cost "
				<< std::fixed << std::setprecision(2) << epoch_duration << "s: "
				<< "Loss = " << std::fixed << std::setprecision(4) << avg_loss
				<< ", Accuracy = " << std::fixed << std::setprecision(2) << accuracy << "%"
				<< " (" << correct << "/" << dataset.size() << ")" << std::endl;
			std::cout << "Current cost: " << std::fixed << std::setprecision(2) << total_elapsed << " sec, "
				<< "Rest time: " << std::fixed << std::setprecision(2) << remaining_time << " sec"
				<< std::endl;
			sum_accuracy += accuracy;
		}
		// 训练结束统计
		auto total_end_time = std::chrono::high_resolution_clock::now();
		auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count() / 1000.0;
		std::cout << "Train finished，totally cost " << std::fixed << std::setprecision(2) << total_duration
			<< " sec (" << std::fixed << std::setprecision(2) << total_duration / 60.0 << " min)"
			<< std::endl;
		return sum_accuracy /= epochs; // 返回平均准确率
	}
	size_t le_net5::predict(const Eigen::VectorXd& output) {
		size_t max_index = 0;
		double max_value = output[0];
		// 找到输出向量中最大值的索引
		for (size_t i = 1; i < output.size(); ++i) {
			if (output[i] > max_value) {
				max_value = output[i];
				max_index = i;
			}
		}
		return max_index;
	}
	void le_net5::save(const std::filesystem::path& path) {
		std::filesystem::create_directory(path);
		conv1_.save(path / "conv1");
		conv2_.save(path / "conv2");
		fc1_.save(path / "fc1.txt");
		fc2_.save(path / "fc2.txt");
		fc3_.save(path / "fc3.txt");
	}
	void le_net5::load(const std::filesystem::path& path) {
		std::filesystem::create_directory(path);
		conv1_.load(path / "conv1");
		conv2_.load(path / "conv2");
		fc1_.load(path / "fc1.txt");
		fc2_.load(path / "fc2.txt");
		fc3_.load(path / "fc3.txt");
	}
	Eigen::VectorXd le_net5::flatten(const std::vector<Eigen::MatrixXd>& matrixs) {
		Eigen::VectorXd result(matrixs.size() * matrixs[0].size());
		size_t index = 0;
		for (const auto& matrix : matrixs) {
			for (size_t r = 0; r < matrix.rows(); r++) {
				for (size_t c = 0; c < matrix.cols(); c++) {
					result(index++) = matrix(r, c);
				}
			}
		}
		return result;
	}
	std::vector<Eigen::MatrixXd> le_net5::counterflatten(const Eigen::VectorXd& vector, size_t channels, size_t rows, size_t cols) {
		std::vector<Eigen::MatrixXd> result;
		size_t index = 0;
		for (size_t i = 0; i < channels; ++i) {
			Eigen::MatrixXd channel(rows, cols);
			for (size_t r = 0; r < rows; ++r) {
				for (size_t c = 0; c < cols; ++c) {
					channel(r, c) = vector[index++];
				}
			}
			result.push_back(channel);
		}
		return result;
	}
	double le_net5::cross_entropy_loss(const Eigen::VectorXd& output, size_t label) {
		// 计算softmax
		Eigen::VectorXd softmax_output = output.array().exp();
		softmax_output /= softmax_output.sum();
		// 计算交叉熵损失：-log(softmax_output[label])
		return -std::log(softmax_output[label] + 1e-8); // 添加小数值防止log(0)
	}

}