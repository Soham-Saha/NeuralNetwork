
abstract class ErrorFunction {
	String name;

	public ErrorFunction(String name) {
		this.name = name;
	}

	public ErrorFunction() {
		this("user_defined");
	}

	public abstract double getError(double[] target, double[] prediction);

	/** Derivative of error function with respect to i^th prediction/output */
	public abstract double derivative(double[] target, double[] prediction, int i);
}
