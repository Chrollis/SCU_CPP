#include "vgg16.hpp"
#include "language_manager.h"
#include <fstream>

namespace chr {
vgg16::vgg16()
    : conv1_(1, 3, 2, 3, 1, activation_function_type::lrelu)
    , // 32*32
    conv2_(2, 3, 2, 1, 1, activation_function_type::lrelu)
    , pool1_(2, 2)
    , // 16*16
    conv3_(2, 3, 4, 1, 1, activation_function_type::lrelu)
    , conv4_(4, 3, 4, 1, 1, activation_function_type::lrelu)
    , pool2_(2, 2)
    , // 8*8
    conv5_(4, 3, 8, 1, 1, activation_function_type::lrelu)
    , conv6_(8, 3, 8, 1, 1, activation_function_type::lrelu)
    , conv7_(8, 3, 8, 1, 1, activation_function_type::lrelu)
    , pool3_(2, 2)
    , // 4*4
    conv8_(8, 3, 16, 1, 1, activation_function_type::lrelu)
    , conv9_(16, 3, 16, 1, 1, activation_function_type::lrelu)
    , conv10_(16, 3, 16, 1, 1, activation_function_type::lrelu)
    , pool4_(2, 2)
    , // 2*2
    conv11_(16, 3, 32, 1, 1, activation_function_type::lrelu)
    , conv12_(32, 3, 32, 1, 1, activation_function_type::lrelu)
    , conv13_(32, 3, 32, 1, 1, activation_function_type::lrelu)
    , pool5_(2, 2)
    , // 1*1
    fc1_(32, 24, activation_function_type::lrelu)
    , fc2_(24, 16, activation_function_type::lrelu)
    , fc3_(16, 10, activation_function_type::lrelu)
{
}
Eigen::VectorXd vgg16::forward(const std::vector<Eigen::MatrixXd>& input)
{
    auto a1 = conv1_.forward(input);
    auto a2 = conv2_.forward(a1);
    auto p1 = pool1_.forward(a2);
    auto a3 = conv3_.forward(p1);
    auto a4 = conv4_.forward(a3);
    auto p2 = pool2_.forward(a4);
    auto a5 = conv5_.forward(p2);
    auto a6 = conv6_.forward(a5);
    auto a7 = conv7_.forward(a6);
    auto p3 = pool3_.forward(a7);
    auto a8 = conv8_.forward(p3);
    auto a9 = conv9_.forward(a8);
    auto a10 = conv10_.forward(a9);
    auto p4 = pool4_.forward(a10);
    auto a11 = conv11_.forward(p4);
    auto a12 = conv12_.forward(a11);
    auto a13 = conv13_.forward(a12);
    auto p5 = pool5_.forward(a13);
    Eigen::VectorXd f = flatten(p5);
    auto a14 = fc1_.forward(f);
    auto a15 = fc2_.forward(a14);
    return fc3_.forward(a15);
}
std::vector<Eigen::MatrixXd> vgg16::backward(size_t label, double learning_rate)
{
    // 按照前向传播的逆序进行反向传播
    auto da15 = fc3_.backward({}, learning_rate, 1, label);
    auto da14 = fc2_.backward(da15, learning_rate);
    auto df = fc1_.backward(da14, learning_rate);
    // 将全连接层的梯度反平铺为卷积层的形状 (32个1x1的特征图)
    auto dp5 = counterflatten(df, 32, 1, 1);
    auto da13 = pool5_.backward(dp5);
    auto da12 = conv13_.backward(da13, learning_rate, 1);
    auto da11 = conv12_.backward(da12, learning_rate);
    auto da10 = conv11_.backward(da11, learning_rate);
    auto dp4 = pool4_.backward(da10);
    auto da9 = conv10_.backward(dp4, learning_rate);
    auto da8 = conv9_.backward(da9, learning_rate);
    auto da7 = conv8_.backward(da8, learning_rate);
    auto dp3 = pool3_.backward(da7);
    auto da6 = conv7_.backward(dp3, learning_rate);
    auto da5 = conv6_.backward(da6, learning_rate);
    auto da4 = conv5_.backward(da5, learning_rate);
    auto dp2 = pool2_.backward(da4);
    auto da3 = conv4_.backward(dp2, learning_rate);
    auto da2 = conv3_.backward(da3, learning_rate);
    auto dp1 = pool1_.backward(da2);
    auto da1 = conv2_.backward(dp1, learning_rate);
    return conv1_.backward(da1, learning_rate);
}
void vgg16::save(const std::filesystem::path& path)
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
    conv3_.save(file);
    conv4_.save(file);
    conv5_.save(file);
    conv6_.save(file);
    conv7_.save(file);
    conv8_.save(file);
    conv9_.save(file);
    conv10_.save(file);
    conv11_.save(file);
    conv12_.save(file);
    conv13_.save(file);
    fc1_.save(file);
    fc2_.save(file);
    fc3_.save(file);
    file.close();
    emit inform(chr::tr("model.io.saved").arg(this->model_type()).arg(path.string()));
}
void vgg16::load(const std::filesystem::path& path)
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
    conv3_.load(file);
    conv4_.load(file);
    conv5_.load(file);
    conv6_.load(file);
    conv7_.load(file);
    conv8_.load(file);
    conv9_.load(file);
    conv10_.load(file);
    conv11_.load(file);
    conv12_.load(file);
    conv13_.load(file);
    fc1_.load(file);
    fc2_.load(file);
    fc3_.load(file);
    file.close();
    emit inform(chr::tr("model.io.loaded").arg(this->model_type()).arg(path.string()));
}

}
