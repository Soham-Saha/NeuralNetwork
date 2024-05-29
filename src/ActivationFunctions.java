//TODO: all functions should have options, like sigmoid will greatly benefit from one.

import java.io.Serializable;

public enum ActivationFunctions implements Serializable {
	SWISH, SIGMOID, LEAKY_RELU, RELU, SELU, TANH;

	public static ActivationFunction get(ActivationFunctions af, double... params) {
		switch (af) {
		case SWISH:
			ActivationFunction swish = new ActivationFunction() {

				@Override
				double derivative(double x) {
					return activationFunction(x) + (1 - activationFunction(x)) / (1 + Math.exp(-x));
				}

				@Override
				double activationFunction(double x) {
					// TODO Auto-generated method stub
					return x / (1 + Math.exp(-x));
				}
			};
			swish.name = "swish";
			return swish;
		case SIGMOID:
			double alpha = params.length > 0 ? params[0] : 1;
			ActivationFunction sigmoid = new ActivationFunction() {
				@Override
				public double activationFunction(double x) {
					return 1 / (1 + Math.exp(-alpha * x));
				}

				@Override
				public double derivative(double x) {
					double k = activationFunction(x);
					return alpha * k * (1 - k);
				}
			};
			sigmoid.name = "sigmoid";
			return sigmoid;
		case LEAKY_RELU:
			ActivationFunction lrelu = new ActivationFunction() {
				double alpha = 0.0001;

				@Override
				double derivative(double x) {
					return x < 0 ? alpha : 1;
				}

				@Override
				double activationFunction(double x) {
					return Math.max(alpha * x, x);
				}
			};
			lrelu.name = "leakyReLU";
			return lrelu;
		case RELU:
			ActivationFunction relu = new ActivationFunction() {

				@Override
				double derivative(double x) {
					return x < 0 ? 0 : 1;
				}

				@Override
				double activationFunction(double x) {
					return Math.max(0, x);
				}
			};
			relu.name = "ReLU";
			return relu;
		case SELU:
			ActivationFunction selu = new ActivationFunction() {
				double a = 1.758;

				@Override
				double derivative(double x) {
					return a * Math.exp(x) - a;
				}

				@Override
				double activationFunction(double x) {
					return a * Math.exp(x);
				}
			};
			selu.name = "SELU";
			return selu;
		case TANH:
			ActivationFunction tanh = new ActivationFunction() {

				@Override
				double derivative(double x) {
					double k = Math.tanh(x);
					return 1 - k * k;
				}

				@Override
				double activationFunction(double x) {
					return Math.tanh(x);
				}
			};
			tanh.name = "tanh";
			return tanh;
		default:
			return null;
		}

	}
}
