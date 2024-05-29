public enum ErrorFunctions {
	MEAN_SQUARED_ERROR,
	/**
	 * Utilizes softmax inside itself. So output prediction will need to use softmax
	 * separately
	 */
	SOFTMAX_CROSSENTROPY;

	public static ErrorFunction getErrorFunction(ErrorFunctions e) {
		switch (e) {
		case MEAN_SQUARED_ERROR:
			ErrorFunction mse = new ErrorFunction("mean_squared_error") {

				@Override
				public double getError(double[] target, double[] prediction) {
					double error = 0;
					for (int i = 0; i < target.length; i++) {
						error += (target[i] - prediction[i]) * (target[i] - prediction[i]);
					}
					error /= target.length;
					return error;
				}

				@Override
				public double derivative(double[] target, double[] prediction, int i) {
					return 2 * (prediction[i] - target[i]) / target.length;
				}
			};
			return mse;
		/**
		 * @author ChatGPT-4o
		 */
		case SOFTMAX_CROSSENTROPY:
			ErrorFunction bces = new ErrorFunction("softmax_crossentropy") {

				@Override
				public double getError(double[] target, double[] prediction) {
					double maxLogit = Double.NEGATIVE_INFINITY;
					for (double x : prediction) {
						if (x > maxLogit) {
							maxLogit = x;
						}
					}

					double sumExp = 0;
					for (double x : prediction) {
						sumExp += Math.exp(x - maxLogit); // Subtract max logit for numerical stability
					}

					double logSumExp = maxLogit + Math.log(sumExp);
					double loss = 0;

					for (int i = 0; i < target.length; i++) {
						loss += target[i] * (logSumExp - prediction[i]);
					}

					return loss;
				}

				@Override
				public double derivative(double[] target, double[] prediction, int i) {
					double maxLogit = Double.NEGATIVE_INFINITY;
					for (double x : prediction) {
						if (x > maxLogit) {
							maxLogit = x;
						}
					}

					double sumExp = 0.0;
					for (double x : prediction) {
						sumExp += Math.exp(x - maxLogit);
					}

					double softmax = Math.exp(prediction[i] - maxLogit) / sumExp;
					return softmax - target[i];
				}
			};
			return bces;
		default:
			return null;
		}

	}
}
