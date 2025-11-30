#include "le_net5.hpp"
#include "language_manager.h"
#include <fstream>

namespace chr {
le_net5::le_net5()
    : conv1_(1, 5, 6, 2, 1, activation_function_type::lrelu)
    , // 输入通道1，输出通道6，5x5卷积核
    pool1_(2, 2)
    , // 2x2最大池化，步长2
    conv2_(6, 5, 16, 0, 1, activation_function_type::lrelu)
    , // 输入通道6，输出通道16，5x5卷积核
    pool2_(2, 2)
    , // 2x2最大池化，步长2
    fc1_(400, 120, activation_function_type::lrelu)
    , // 2x2最大池化，步长2
    fc2_(120, 84, activation_function_type::lrelu)
    , // 120->84全连接
    fc3_(84, 10, activation_function_type::lrelu)
{ // 84->10全连接(输出层)
}
Eigen::VectorXd le_net5::forward(const std::vector<Eigen::MatrixXd>& input)
{
    auto a1 = conv1_.forward(input);
    auto p1 = pool1_.forward(a1);
    auto a2 = conv2_.forward(p1);
    auto p2 = pool2_.forward(a2);
    Eigen::VectorXd f = flatten(p2); // 展平为向量
    auto a3 = fc1_.forward(f);
    auto a4 = fc2_.forward(a3);
    return fc3_.forward(a4);
}
std::vector<Eigen::MatrixXd> le_net5::backward(size_t label, double learning_rate)
{
    auto da4 = fc3_.backward({}, learning_rate, 1, label);
    auto da3 = fc2_.backward(da4, learning_rate);
    auto df = fc1_.backward(da3, learning_rate);
    auto dp2 = counterflatten(df, 16, 5, 5); // 反展平为张量
    auto da2 = pool2_.backward(dp2);
    auto dp1 = conv2_.backward(da2, learning_rate, 1);
    auto da1 = pool1_.backward(dp1);
    return conv1_.backward(da1, learning_rate);
}
void le_net5::save(const std::filesystem::path& path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(chr::tr("error.file.save_failed").arg(path.string()).toStdString());
    }
    // 写入魔数 1128
    uint32_t magic_number = 1128;
    file.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));
    // 写入模型类型名
    std::string model_type = this->model_type();
    uint32_t type_length = model_type.length();
    file.write(reinterpret_cast<const char*>(&type_length), sizeof(type_length));
    file.write(model_type.c_str(), type_length);
    // 保存各层参数
    conv1_.save(file);
    conv2_.save(file);
    fc1_.save(file);
    fc2_.save(file);
    fc3_.save(file);
    file.close();
    emit inform(chr::tr("model.io.saved").arg(this->model_type()).arg(path.string()));
}
void le_net5::load(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(chr::tr("error.file.load_failed").arg(path.string()).toStdString());
    }
    // 读取并验证魔数
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 1128) {
        throw std::runtime_error(chr::tr("error.file.invalid_magic_number").toStdString());
    }
    // 读取并验证模型类型
    uint32_t type_length;
    file.read(reinterpret_cast<char*>(&type_length), sizeof(type_length));
    std::string model_type(type_length, ' ');
    file.read(&model_type[0], type_length);
    if (model_type != this->model_type()) {
        throw std::runtime_error(chr::tr("error.file.model_type_mismatch").arg(this->model_type()).arg(model_type).toStdString());
    }
    // 加载各层参数
    conv1_.load(file);
    conv2_.load(file);
    fc1_.load(file);
    fc2_.load(file);
    fc3_.load(file);
    file.close();
    emit inform(chr::tr("model.io.loaded").arg(this->model_type()).arg(path.string()));
}
}
