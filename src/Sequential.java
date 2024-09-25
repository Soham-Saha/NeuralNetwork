import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

//TODO: Softmax built inside softmax_crossentropy. need to be put as a separate layer.

public class Sequential implements Serializable {
	private static final long serialVersionUID = 1493093680782809727L;
	Layer[] layers;
	FullyConnectedLayer[] edges;
	ConversionRule inputConversionRule = x -> x;

	public Sequential(Layer... layers) {
		this.layers = layers;
		connectLayers();
		initializeAll(); // TODO: design choice; Would user want to implement this every time?
	}

	public static double[] softmax(double[] prediction) {
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

		for (int i = 0; i < prediction.length; i++) {
			prediction[i] = Math.exp(prediction[i] - maxLogit) / sumExp;
		}
		return prediction;
	}

	public Sequential(int... nodeCounts) {
		this.layers = new Layer[nodeCounts.length];
		this.layers[0] = new Layer(nodeCounts[0], (ActivationFunction) null);
		for (int i = 1; i < nodeCounts.length; i++) {
			this.layers[i] = new Layer(nodeCounts[i]);
		}
		connectLayers();
		initializeAll(); // TODO: design choice; Would user want to implement this every time?
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		Sequential s = new Sequential(this.layers.length);
		s.layers = this.layers.clone();
		s.edges = this.edges.clone();
		s.inputConversionRule = this.inputConversionRule;
		return s;
	}

	public void describe() {
		for (Layer l : layers) {
			if (l.activation != null) {
				System.out.println(l.numNodes + " " + l.activation.name);
			} else {
				System.out.println(l.numNodes + " N/A");
			}
		}

	}

	public double[] predict(double... input) {
		return predict(true, input);
	}

	public double[] predict(boolean applyConversionFirst, double... input) {
		if (applyConversionFirst) {
			input = inputConversionRule.run(input);
		}
		if (layers[0].numNodes != input.length) {
			throw new IllegalArgumentException("Input shape doesn't match network input shape.");
		}
		double[] currentArray = input;
		for (int i = 0; i <= layers.length - 2; i++) {
			double[] newArray = new double[layers[i + 1].numNodes];
			double[][] edgeWeightsNow = edges[i].edgeWeights;
			double[] biasesNow = layers[i + 1].getBiases();
			for (int x = 0; x < newArray.length; x++) {
				for (int y = 0; y < currentArray.length; y++) {
					newArray[x] += currentArray[y] * edgeWeightsNow[y][x];
				}
				newArray[x] += biasesNow[x];
				newArray[x] = layers[i + 1].activation.activationFunction(newArray[x]);
			}
			currentArray = newArray;
		}
		return currentArray;
	}

	public double getWeight(int layer, int from, int to) {
		return edges[layer].edgeWeights[from][to];
	}

	public void dumpData() {
		System.out.println("Weights: ");
		for (int i = 0; i < edges.length; i++) {
			System.out.println("At layer between " + i + " and " + (i + 1) + " " + Arrays.deepToString(edges[i].edgeWeights));
		}
		System.out.println("Biases: ");
		for (int i = 0; i < layers.length; i++) {
			System.out.println("At layer " + i + " " + Arrays.toString(layers[i].getBiases()));
		}
	}

	public void setWeight(int layer, int from, int to, double newValue) {
		edges[layer].edgeWeights[from][to] = newValue;
	}

	public void changeWeightBy(int layer, int from, int to, double change) {
		edges[layer].edgeWeights[from][to] += change;
	}

	public void changeBiasBy(int layer, int index, double change) {
		layers[layer].biases[index] += change;
	}

	private void connectLayers() {
		edges = new FullyConnectedLayer[layers.length - 1];
		for (int i = 0; i < layers.length - 1; i++) {
			edges[i] = new FullyConnectedLayer(layers[i], layers[i + 1]);
		}
	}

	public void initializeAll() {
		initializeBiases();
		initializeWeights();
	}

	private void initializeWeights() {
		Random r = new Random();
		for (FullyConnectedLayer fcl : edges) {
			for (int i = 0; i < fcl.edgeWeights.length; i++) {
				for (int j = 0; j < fcl.edgeWeights[i].length; j++) {
					fcl.edgeWeights[i][j] = r.nextGaussian() * Math.sqrt(2 / (double) fcl.layerIn.numNodes);
				}
			}
		}
	}

	public void initializeBiases() {
		for (Layer l : layers) {
			for (int i = 0; i < l.numNodes; i++) {
				l.biases[i] = 0;
			}
		}
	}
}