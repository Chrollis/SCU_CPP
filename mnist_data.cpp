#include "mnist_data.hpp"
#include <fstream>

namespace chr {
	bool mnist_data::is_legal() const {
		if (image_.rows() != 28 || image_.cols() != 28) return 0; // 检查是否为28x28尺寸
		return 1;
	}
	// 字节序转换函数(大端转小端)
	unsigned mnist_data::swap_endian(unsigned val) {
		val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
		return (val << 16) | (val >> 16);
	}
	// 验证MNIST文件格式并返回数据项数量
	unsigned mnist_data::check_mnist_file(std::ifstream& mnist_images, std::ifstream& mnist_labels) {
		if (!mnist_images.is_open()) {
			throw std::runtime_error("failed to open MNIST image file");
		}
		if (!mnist_labels.is_open()) {
			throw std::runtime_error("failed to open MNIST label file");
		}
		unsigned magic = 0;
		unsigned items = 0, labels = 0;
		// 检查图像文件魔数
		mnist_images.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2051) { // MNIST图像文件魔数应为2051
			throw std::runtime_error("not MNIST image file");
		}
		// 检查标签文件魔数
		mnist_labels.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2049) { // MNIST标签文件魔数应为2049
			throw std::runtime_error("not MNIST label file");
		}
		// 读取数据项数量
		mnist_images.read(reinterpret_cast<char*>(&items), 4);
		items = swap_endian(items);
		mnist_labels.read(reinterpret_cast<char*>(&labels), 4);
		labels = swap_endian(labels);
		if (items != labels) { // 图像和标签数量必须匹配
			throw std::runtime_error("image file and label file are not a pair");
		}
		return items;
	}
	// 从MNIST文件读取数据
	std::vector<mnist_data> mnist_data::obtain_data(const std::filesystem::path& mnist_image_path, const std::filesystem::path& mnist_label_path, size_t offset, size_t size) {
		std::ifstream mnist_images(mnist_image_path, std::ios::binary);
		std::ifstream mnist_labels(mnist_label_path, std::ios::binary);
		size_t items = 0;
		try {
			items = check_mnist_file(mnist_images, mnist_labels); // 验证文件格式
		}
		catch (const std::exception& e) {
			std::cerr << "[mnist_data::obtain_data]: " << e.what() << std::endl;
			return {};
		}
		if (offset >= items) return {}; // 偏移量不能超过数据集大小
		std::vector<mnist_data> batch;
		unsigned rows = 0, cols = 0;
		// 读取图像尺寸
		mnist_images.read(reinterpret_cast<char*>(&rows), 4);
		rows = swap_endian(rows);
		mnist_images.read(reinterpret_cast<char*>(&cols), 4);
		cols = swap_endian(cols);
		// 跳转到指定偏移量
		mnist_images.seekg(offset * rows * cols, std::ios::cur);
		mnist_labels.seekg(offset, std::ios::cur);
		// 读取指定数量的数据
		for (size_t i = 0; i < size && i + offset < items; i++) {
			Eigen::MatrixXd image(rows, cols);
			std::vector<unsigned char> pixels(rows * cols);
			size_t label = 0;
			// 读取图像像素和标签
			mnist_images.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(rows) * cols);
			mnist_labels.read(reinterpret_cast<char*>(&label), 1);
			// 将像素数据转换为矩阵(二值化)
			for (unsigned m = 0; m < rows; m++) {
				for (unsigned n = 0; n < cols; n++) {
					image(m, n) = pixels[static_cast<size_t>(m) * cols + n] ? 1.0 : 0.0;
				}
			}
			mnist_data piece(std::move(image), label);
			batch.push_back(std::move(piece));
		}
		return batch;
	}
}