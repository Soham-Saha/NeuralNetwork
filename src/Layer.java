import java.io.Serializable;

public class Layer implements Serializable {
	private static ActivationFunctions defaultActivationFunction = ActivationFunctions.RELU;
	int numNodes;
	double[] biases;
	ActivationFunction activation;

	public Layer(int numNodes) {
		this(numNodes, defaultActivationFunction);
	}

	public Layer(int numNodes, ActivationFunctions a) {
		this(numNodes, ActivationFunctions.get(a));
	}

	public Layer(int numNodes, ActivationFunction a) {
		this.numNodes = numNodes;
		this.biases = new double[numNodes];
		this.activation = a;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		Layer l = new Layer(numNodes);
		l.biases = getBiases();
		l.activation = this.activation;
		return l;
	}

	public double[] getBiases() {
		return biases.clone();
	}
}
