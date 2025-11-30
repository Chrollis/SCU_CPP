#include "full_connect_layer.hpp"
#include "language_manager.h"
#include <fstream>
#include <random>

namespace chr {
full_connect_layer::full_connect_layer(size_t in_size, size_t out_size, activation_function_type activate_type)
    : in_size_(in_size)
    , out_size_(out_size)
    , afunc_(activate_type)
{
    initialize_weights();
}
Eigen::VectorXd full_connect_layer::forward(const Eigen::VectorXd& input)
{
    input_ = input;
    // 线性计算：output = weights * input + biases
    linear_outcome_ = weights_ * input_ + biases_;
    feature_vector_ = linear_outcome_.unaryExpr([this](double a) { return afunc_(a); });
    return feature_vector_;
}
Eigen::VectorXd full_connect_layer::backward(const Eigen::VectorXd& gradient, double learning_rate, bool is_output_layer, size_t label)
{
    if (is_output_layer) {
        // 如果是输出层，计算交叉熵损失梯度
        Eigen::VectorXd target = Eigen::VectorXd::Zero(out_size_);
        if (label >= 0 && label < out_size_) {
            target(label) = 1.0; // 创建one-hot标签
        }
        // 输出层梯度 = 预测值 - 真实值
        gradient_ = feature_vector_ - target;
    } else {
        // 隐藏层梯度 = 上层梯度 × 激活函数导数
        gradient_ = gradient.cwiseProduct(linear_outcome_.unaryExpr([this](double a) { return afunc_[a]; }));
    }
    Eigen::VectorXd next_gradient = weights_.transpose() * gradient_;
    weights_update(learning_rate);
    return next_gradient;
}
void full_connect_layer::weights_update(double learning_rate)
{
    // 权重更新：weights -= 学习率 * 梯度 * 输入转置
    weights_ -= learning_rate * gradient_ * input_.transpose();
    // 偏置更新：biases -= 学习率 * 梯度
    biases_ -= learning_rate * gradient_;
}
void full_connect_layer::save(std::ostream& file) const
{
    // 保存全连接层基本信息
    uint32_t in_size = in_size_;
    uint32_t out_size = out_size_;
    file.write(reinterpret_cast<const char*>(&in_size), sizeof(in_size));
    file.write(reinterpret_cast<const char*>(&out_size), sizeof(out_size));
    // 保存权重矩阵
    for (size_t i = 0; i < out_size_; ++i) {
        for (size_t j = 0; j < in_size_; ++j) {
            double weight = weights_(i, j);
            file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
    }
    // 保存偏置向量
    for (size_t i = 0; i < out_size_; ++i) {
        double bias = biases_(i);
        file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    }
}
void full_connect_layer::load(std::istream& file)
{
    // 读取全连接层基本信息
    uint32_t in_size, out_size;
    file.read(reinterpret_cast<char*>(&in_size), sizeof(in_size));
    file.read(reinterpret_cast<char*>(&out_size), sizeof(out_size));
    // 验证参数是否匹配
    if (in_size != in_size_ || out_size != out_size_) {
        throw std::runtime_error(chr::tr("error.fully_connected.parameter_mismatch").toStdString());
    }
    // 读取权重矩阵
    for (size_t i = 0; i < out_size_; ++i) {
        for (size_t j = 0; j < in_size_; ++j) {
            double weight;
            file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            weights_(i, j) = weight;
        }
    }
    // 读取偏置向量
    for (size_t i = 0; i < out_size_; ++i) {
        double bias;
        file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        biases_(i) = bias;
    }
}
void full_connect_layer::initialize_weights()
{
    weights_.resize(out_size_, in_size_);
    biases_.resize(out_size_);
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0.0, 0.01); // 使用小方差高斯分布
    for (size_t i = 0; i < out_size_; ++i) {
        for (size_t j = 0; j < in_size_; ++j) {
            weights_(i, j) = distribution(generator);
        }
        biases_(i) = distribution(generator);
    }
}
}
