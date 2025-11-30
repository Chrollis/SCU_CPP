#ifndef LE_NET5_HPP
#define LE_NET5_HPP

#include "cnn_base.h"
#include "convolve_layer.hpp"
#include "full_connect_layer.hpp"
#include "pool_layer.hpp"

namespace chr {
class le_net5 : public cnn_base {
private:
    // 网络层定义(LeNet-5结构)
    convolve_layer conv1_; // 第一卷积层
    pool_layer pool1_; // 第一池化层
    convolve_layer conv2_; // 第二卷积层
    pool_layer pool2_; // 第二池化层
    full_connect_layer fc1_; // 第一全连接层
    full_connect_layer fc2_; // 第二全连接层
    full_connect_layer fc3_; // 第三全连接层(输出层)

public:
    le_net5();
    Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(size_t label, double learning_rate) override;
    void save(const std::filesystem::path& path) override;
    void load(const std::filesystem::path& path) override;
    std::string model_type() const override { return "LeNet-5"; }
};
}

#endif // !LE_NET5_HPP
