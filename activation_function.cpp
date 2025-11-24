#include "activation_function.hpp"

namespace chr {
	void activation_function::change_function(activation_function_type type) {
		switch (type)
		{
		case activation_function_type::sigmoid:
			function = [this](double x) {return sigmoid(x); };
			derivative = [this](double x) {return sigmoid_derivative(x); };
			break;
		case activation_function_type::tanh:
			function = [this](double x) {return tanh(x); };
			derivative = [this](double x) {return tanh_derivative(x); };
			break;
		case activation_function_type::relu:
			function = [this](double x) {return relu(x); };
			derivative = [this](double x) {return relu_derivative(x); };
			break;
		case activation_function_type::lrelu:
			function = [this](double x) {return lrelu(x); };
			derivative = [this](double x) {return lrelu_derivative(x); };
			break;
		default:
			break;
		}
	}
}