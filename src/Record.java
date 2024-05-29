import java.io.Serializable;

public class Record implements Serializable {
	double[] input;
	double[] output;

	public Record(double[] input, double[] output) {
		this.input = input;
		this.output = output;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return new Record(input.clone(), output.clone());
	}
}
