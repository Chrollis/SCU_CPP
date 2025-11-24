#ifndef MNIST_DATA_HPP
#define MNIST_DATA_HPP

#include <filesystem>
#include "image_process.hpp"

namespace chr {
	// MNIST数据集数据类，用于存储手写数字图像和标签
	class mnist_data {
	private:
		Eigen::MatrixXd image_; // 28x28的图像矩阵，像素值归一化为0或1
		size_t label_; // 数字标签(0-9)
	public:
		mnist_data(const Eigen::MatrixXd& image, size_t label) :image_(image), label_(label) { ; }
		mnist_data(Eigen::MatrixXd&& image, size_t label) :image_(image), label_(label) { ; }
		const Eigen::MatrixXd& image() const { return image_; } // 获取图像矩阵
		cv::Mat cv_image() const { return image_process::matrix_to_image(image_); } // 转换为OpenCV图像格式
		size_t label() const { return label_; } // 获取标签
		bool is_legal() const; // 检查数据是否合法(28x28尺寸)
	private:
		static unsigned swap_endian(unsigned val); // 字节序转换(大端转小端)
		static unsigned check_mnist_file(std::ifstream& mnist_images, std::ifstream& mnist_labels); // 验证MNIST文件格式
	public:
		// 从MNIST文件读取数据
		static std::vector<mnist_data> obtain_data(const std::filesystem::path& mnist_image_path, const std::filesystem::path& mnist_label_path, size_t offset = 0, size_t size = 60000);
	};
}

#endif // !MNIST_DATA_HPP
