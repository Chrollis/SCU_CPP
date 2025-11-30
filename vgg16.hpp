#ifndef VGG16_HPP
#define VGG16_HPP

#include "cnn_base.h"
#include "convolve_layer.hpp"
#include "full_connect_layer.hpp"
#include "pool_layer.hpp"

namespace chr {
class vgg16 : public cnn_base {
private:
    // 网络层定义(VGG16结构)
    // 块 1
    convolve_layer conv1_;
    convolve_layer conv2_;
    pool_layer pool1_;
    // 块 2
    convolve_layer conv3_;
    convolve_layer conv4_;
    pool_layer pool2_;
    // 块 3
    convolve_layer conv5_;
    convolve_layer conv6_;
    convolve_layer conv7_;
    pool_layer pool3_;
    // 块 4
    convolve_layer conv8_;
    convolve_layer conv9_;
    convolve_layer conv10_;
    pool_layer pool4_;
    // 块 5
    convolve_layer conv11_;
    convolve_layer conv12_;
    convolve_layer conv13_;
    pool_layer pool5_;
    // 分类器
    full_connect_layer fc1_;
    full_connect_layer fc2_;
    full_connect_layer fc3_;

public:
    vgg16();
    Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(size_t label, double learning_rate) override;
    void save(const std::filesystem::path& path) override;
    void load(const std::filesystem::path& path) override;
    std::string model_type() const override { return "VGG16"; }
};
}

#endif // !VGG16_HPP
