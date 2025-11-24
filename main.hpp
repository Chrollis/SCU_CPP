#ifndef MAIN_HPP
#define MAIN_HPP

#include "le_net5.hpp"

// 测试MNIST数据读取功能
static void test_read() {
	auto b = chr::mnist_data::obtain_data(
		"MNIST_data/train-images.idx3-ubyte",
		"MNIST_data/train-labels.idx1-ubyte",
		1, 100 // 从第1个样本开始，读取100个样本
	);
	for (int i = 0; i < b.size(); i++) {
		if (!b[i].is_legal()) exit(-1); // 检查数据合法性
		cv::Mat img;
		cv::resize(b[i].cv_image(), img, cv::Size(160, 160));
		cv::imshow(std::to_string(b[i].label()), img);
		cv::waitKey(1000);
		cv::destroyWindow(std::to_string(b[i].label()));
	}
}
// 测试图像处理功能
static void test_image_process() {
	cv::Mat img = cv::imread("test.jpg");
	cv::imshow("test", img);
	cv::waitKey(1000);
	cv::destroyWindow("test");
    // 处理图像中的数字
	auto b = chr::image_process::process_image(img);
	for (int i = 0; i < b.size(); i++) {
		cv::Mat img2;
		cv::resize(chr::image_process::matrix_to_image(b[i]), img2, cv::Size(160, 160));
		cv::imshow("test", img2);
		cv::waitKey(1000);
		cv::destroyWindow("test");
	}
}
// 测试模型功能
static void test_model() {
    auto b = chr::mnist_data::obtain_data(
        "MNIST_data/train-images.idx3-ubyte",
        "MNIST_data/train-labels.idx1-ubyte",
        0, 1
    );
	cv::Mat img;
	cv::resize(b[0].cv_image(), img, cv::Size(160, 160));
	cv::imshow(std::to_string(b[0].label()), img);
	try {
		chr::le_net5 model; // 创建LeNet-5模型
		model.load("trained_lenet5_model"); // 加载训练好的模型
		auto v = model.forward({ b[0].image() }); // 前向传播
		std::cout << v << std::endl; // 输出预测结果
		auto m = model.backward(b[0].label(), 0.001); // 反向传播
		for (const auto& c : m) {
			std::cout << c << std::endl;  // 输出梯度
		}
		model.save("trained_lenet5_model"); // 保存模型
	}
	catch (const std::exception& e) {
		std::cout << "[test_model]: " << e.what() << std::endl;
	}
}
// 训练函数
static void train(size_t batch_size, size_t epochs, double target, double learning_rate = 0.001) {
    try {
        Eigen::setNbThreads(omp_get_max_threads()); // 设置Eigen使用多线程
        std::cout << "Read the train data..." << std::endl;
        auto train = chr::mnist_data::obtain_data(
            "MNIST_data/train-images.idx3-ubyte",
            "MNIST_data/train-labels.idx1-ubyte"
        );
        chr::le_net5 model; // 创建LeNet-5模型
        std::cout << "Load the LeNet-5 model..." << std::endl;
        model.load("trained_lenet5_model"); // 加载已有模型
        std::cout << "Start training..." << std::endl;
        std::cout << "Samples: " << train.size() << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Learning rate: " << learning_rate << std::endl;
        // 将训练数据分批
        std::vector<std::vector<chr::mnist_data>> train_batches;
        std::vector<chr::mnist_data> batch;
        for (auto& data : train) {
            batch.push_back(std::move(data));
            if (batch.size() >= batch_size) {
                train_batches.push_back(std::move(batch));
                batch.clear();
            }
        }
        if (batch.size() > 0) {
            train_batches.push_back(std::move(batch));
            batch.clear();
        }
        train.clear();
        double accuracy = 0.0;
        do {
            static size_t k = 0;
            // 训练当前批次
            accuracy = model.train(train_batches[k], epochs, learning_rate);
            k = (++k) % train_batches.size(); // 循环使用批次
            std::cout << "Save the model..." << std::endl;
            model.save("trained_lenet5_model"); // 保存模型
        } while (accuracy <= target); // 达到目标准确率后停止
        // 在测试集上评估模型
        std::cout << "Read the test data..." << std::endl;
        auto test = chr::mnist_data::obtain_data(
            "MNIST_data/t10k-images.idx3-ubyte",
            "MNIST_data/t10k-labels.idx1-ubyte"
        );
        std::cout << "Evaluate on the test set..." << std::endl;
        size_t correct = 0;
        for (size_t i = 0; i < test.size(); ++i) {
            auto output = model.forward({ test[i].image() });
            size_t predicted = model.predict(output);
            if (predicted == test[i].label()) {
                correct++;
            }
            double accuracy = static_cast<double>(correct) / test.size();
            std::cout << "Test accuracy: " << std::to_string(accuracy * 100) << "% ("
                << correct << "/" << test.size() << ")" << std::endl;
        }
        std::cout << "Acomplished!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "[train]: " << e.what() << std::endl;
    }
}

#endif // !MAIN_HPP