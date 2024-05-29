import java.io.Serializable;

public abstract class ActivationFunction implements Serializable {
	String name = "user_defined";

	abstract double activationFunction(double x);

	abstract double derivative(double x);

}
