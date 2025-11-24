#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <cmath>
#include <functional>

namespace chr {
	enum class activation_function_type {
		sigmoid, // S型函数
		tanh, // 双曲正切函数  
		relu, // 修正线性单元
		lrelu, // 泄漏修正线性单元
	};

	class activation_function {
	private:
		std::function<double(double)> function; // 激活函数
		std::function<double(double)> derivative; // 激活函数的导数
	public:
		activation_function(activation_function_type type) { change_function(type); }
		double operator()(double x) { return function(x); } // 前向传播：计算激活值
		double operator[](double x) { return derivative(x); } // 反向传播：计算导数值
		void change_function(activation_function_type type); // 切换激活函数类型
		// 各种激活函数实现
		double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
		double sigmoid_derivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
		double tanh(double x) { return std::tanh(x); }
		double tanh_derivative(double x) { return 1 / pow(cosh(x), 2); }
		double relu(double x) { return x > 0 ? x : 0; }
		double relu_derivative(double x) { return x > 0 ? 1 : 0; }
		double lrelu(double x) { return x > 0 ? x : 0.01 * x; }
		double lrelu_derivative(double x) { return x > 0 ? 1 : 0.01; }
	};
}

#endif // !ACTIVATION_FUNCTION_HPP
