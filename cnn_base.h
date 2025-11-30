#ifndef CNN_BASE_H
#define CNN_BASE_H

#include "mnist_data.hpp"
#include <QObject>

namespace chr {

class cnn_base : public QObject {
    Q_OBJECT
public:
    virtual ~cnn_base() = default;
    virtual Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input) = 0;
    virtual std::vector<Eigen::MatrixXd> backward(size_t label, double learning_rate) = 0;
    double train(const std::vector<mnist_data>& dataset, size_t epochs, double learning_rate, bool show_detail = 0);
    size_t predict(const Eigen::VectorXd& output); // 预测函数
    static std::string model_type_of(const std::filesystem::path& path);
    virtual void save(const std::filesystem::path& path) = 0;
    virtual void load(const std::filesystem::path& path) = 0;
    virtual std::string model_type() const = 0;

signals:
    void inform(const QString& output);
    void train_details(double progress, double loss, size_t correct, size_t total);

protected:
    Eigen::VectorXd flatten(const std::vector<Eigen::MatrixXd>& matrixs); // 展平操作(多维转一维)
    std::vector<Eigen::MatrixXd> counterflatten(const Eigen::VectorXd& vector, size_t channels, size_t rows, size_t cols); // 反展平操作(一维转多维)
    double cross_entropy_loss(const Eigen::VectorXd& output, size_t label); // 交叉熵损失函数
};
}

#endif // CNN_BASE_H
