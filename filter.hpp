#ifndef FILTER_HPP
#define FILTER_HPP

#include <Eigen/Dense>
#include <filesystem>

namespace chr {
	// 卷积核过滤器类
	class filter {
	private:
		size_t core_size_; // 卷积核尺寸
		size_t channels_; // 输入通道数
	public:
		double bias; // 偏置项
		std::vector<Eigen::MatrixXd> kernels; // 卷积核矩阵集合
	public:
		filter(size_t channels, size_t core_size) :core_size_(core_size), channels_(channels) { initialize_gausz(0.01); } // 默认使用高斯初始化
		void initialize_gausz(double stddev); // 高斯初始化
		void initialize_xavier(size_t input_size); // Xavier初始化
		void initialize_He(size_t input_size); // He初始化
		// 模型保存和加载
		void save(const std::filesystem::path& path) const;
		void load(const std::filesystem::path& path);
	};
}

#endif // !FILTER_HPP
