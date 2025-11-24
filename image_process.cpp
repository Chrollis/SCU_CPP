#include "image_process.hpp"

namespace chr {
	namespace image_process {
		// 将Eigen矩阵转换为OpenCV图像(灰度图)
		cv::Mat matrix_to_image(const Eigen::MatrixXd& matrix) {
			int rows = static_cast<int>(matrix.rows()), cols = static_cast<int>(matrix.cols());
			cv::Mat image(rows, cols, CV_8UC1); // 创建8位无符号单通道图像
			for (int m = 0; m < rows; m++) {
				unsigned char* ptr = image.ptr<unsigned char>(m);
				for (int n = 0; n < cols; n++) {
					ptr[n] = static_cast<unsigned char>(matrix(m, n) * 255); // 将0-1值转换为0-255
				}
			}
			return image;
		}
		// 处理单个数字图像：填充、缩放、二值化
		Eigen::MatrixXd process_digit(const cv::Mat& digit_mat) {
			cv::Mat copy = digit_mat.clone();
			apply_padding(copy); // 添加填充
			cv::Mat resized;
			cv::resize(copy, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA); // 缩放到28x28
			Eigen::MatrixXd eigen_digit(28, 28);
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					eigen_digit(i, j) = resized.at<unsigned char>(i, j) ? 1.0 : 0.0; // 二值化
				}
			}
			return eigen_digit;
		}
		// 为图像添加填充，使其接近正方形并添加边界
		void apply_padding(cv::Mat& img) {
			int top, bottom, left, right;
			// 根据宽高比计算填充大小
			if (img.rows > img.cols) {
				left = (img.rows - img.cols) / 2;
				right = img.rows - img.cols - left;
				top = bottom = 0;
			}
			else {
				top = (img.cols - img.rows) / 2;
				bottom = img.cols - img.rows - top;
				left = right = 0;
			}
			// 添加填充使图像变为正方形
			cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
			// 添加额外边界
			int border_width = img.rows / 5;
			cv::copyMakeBorder(img, img, border_width, border_width, border_width, border_width, cv::BORDER_CONSTANT, cv::Scalar(0));
		}
		// 检查矩形区域是否为有效的数字区域
		bool is_valid_digit_region(const cv::Rect& rect, const cv::Size& image_size) {
			int min_size = static_cast<int>(std::min(image_size.width, image_size.height) * 0.1); // 最小尺寸阈值
			return (rect.width > min_size && rect.height > min_size);
		}
		// 处理整张图像，提取所有数字区域
		std::vector<Eigen::MatrixXd> process_image(const cv::Mat& digits_mat) {
			std::vector<Eigen::MatrixXd> digits;
			cv::Mat processed = binarize_image(digits_mat); // 二值化
			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Vec4i> hierarchy;
			// 查找轮廓
			cv::findContours(processed, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
			for (size_t i = 0; i < contours.size(); i++) {
				cv::Rect bounding_rect = cv::boundingRect(contours[i]); // 获取边界矩形
				if (is_valid_digit_region(bounding_rect, digits_mat.size())) {
					cv::Mat digit_mat = processed(bounding_rect); // 提取数字区域
					Eigen::MatrixXd digit = process_digit(digit_mat); // 处理单个数字
					if (digit.size() > 0) {
						digits.push_back(digit);
					}
				}
			}
			return digits;
		}
		// 图像二值化处理
		cv::Mat binarize_image(const cv::Mat& src_img) {
			cv::Mat gray;
			// 转换为灰度图(如果是彩色图)
			if (src_img.channels() == 3) {
				cv::cvtColor(src_img, gray, cv::COLOR_BGR2GRAY);
			}
			else {
				gray = src_img.clone();
			}
			cv::Mat binary;
			cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY_INV); // 反二值化(黑底白字)
			return binary;
		}
	}
}