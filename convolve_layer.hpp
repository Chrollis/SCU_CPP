#ifndef CONVOLVE_LAYER_HPP
#define CONVOLVE_LAYER_HPP

#include "activation_function.hpp"
#include "filter.hpp"
#include <Eigen/Dense>

namespace chr {
class convolve_layer {
private:
    size_t in_channels_; // 输入通道数
    size_t kernel_size_; // 卷积核尺寸
    size_t out_channels_; // 输出通道数(过滤器数量)
    size_t padding_; // 填充大小
    size_t stride_; // 步长
    activation_function afunc_; // 激活函数
    std::vector<filter> filters_; // 过滤器集合
    std::vector<Eigen::MatrixXd> input_; // 输入数据(带填充)
    std::vector<Eigen::MatrixXd> feature_map_; // 特征图(激活后)
    std::vector<Eigen::MatrixXd> convolve_outcome_; // 卷积结果(激活前)
public:
    convolve_layer(size_t in_channels, size_t kernel_size, size_t out_channels, size_t padding = 0, size_t stride = 1, activation_function_type activate_type = activation_function_type::relu);
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input);
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& gradient, double learning_rate, bool is_last_conv = false);
    void weights_update(double learning_rate, const std::vector<Eigen::MatrixXd>& gradient);
    void save(std::ostream& file) const;
    void load(std::istream& file);

private:
    std::vector<Eigen::MatrixXd> padding(const std::vector<Eigen::MatrixXd>& input, size_t circle_num, double fill_num = 0.0) const;
    Eigen::MatrixXd convolve(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel, size_t stride) const;
    std::vector<Eigen::MatrixXd> apply_activation_derivative(const std::vector<Eigen::MatrixXd>& gradient); // 应用激活函数导数
    std::vector<Eigen::MatrixXd> remove_padding(const std::vector<Eigen::MatrixXd>& input, size_t padding);
};
}

#endif // !CONVOLVE_LAYER_HPP
