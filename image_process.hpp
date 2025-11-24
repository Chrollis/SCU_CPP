#ifndef IMAGE_PROCESS_HPP
#define IMAGE_PROCESS_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace chr {
	namespace image_process {
		cv::Mat matrix_to_image(const Eigen::MatrixXd& matrix);
		Eigen::MatrixXd process_digit(const cv::Mat& digit_mat);
		void apply_padding(cv::Mat& img);
		bool is_valid_digit_region(const cv::Rect& rect, const cv::Size& image_size);
		std::vector<Eigen::MatrixXd> process_image(const cv::Mat& digits_mat);
		cv::Mat binarize_image(const cv::Mat& src_img);
	}
}

#endif // !IMAGE_PROCESS_HPP
