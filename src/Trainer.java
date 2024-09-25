//TODO: Softmax built inside softmax_crossentropy. need to be put as a separate layer.

public class Trainer {
	Sequential neuralNet;
	ErrorFunction errorfunc;
	double learningRate;
	DataSet traindataset;

	public Trainer(Sequential neuralNet, ErrorFunctions errorfunc, double learningRate) {
		this(neuralNet, ErrorFunctions.getErrorFunction(errorfunc), learningRate);
	}

	public Trainer(Sequential neuralNet, ErrorFunction errorfunc, double learningRate) {
		this.neuralNet = neuralNet;
		this.errorfunc = errorfunc;
		this.learningRate = learningRate;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		Trainer k = new Trainer(null, errorfunc, learningRate);
		k.neuralNet = (Sequential) this.neuralNet.clone();
		k.traindataset = (DataSet) this.traindataset.clone();
		return k;
	}

	private void trainSample(double[] input, double[] target) {
		Layer[] layers = neuralNet.layers;
		FullyConnectedLayer[] edges = neuralNet.edges;
		if (layers[0].numNodes != input.length) {
			throw new IllegalArgumentException("Input shape doesn't match network input shape.");
		}
		if (layers[layers.length - 1].numNodes != target.length) {
			throw new IllegalArgumentException("Output shape doesn't match network output shape.");
		}
		// Exactly same as predict, only logging outputs at each layer, before they get
		// through the activation
		double[][] layerOutputs = new double[layers.length][];
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
			}
			layerOutputs[i + 1] = newArray;
			for (int x = 0; x < newArray.length; x++) {
				newArray[x] = layers[i + 1].activation.activationFunction(newArray[x]);
			}
			currentArray = newArray;
		}

		double[] nextLayerGrads = new double[layers[layers.length - 1].numNodes];// grads of layer after layerCt layer
		for (int i = 0; i < nextLayerGrads.length; i++) {
			// Derivative of error with respect to i^th output
			nextLayerGrads[i] = errorfunc.derivative(target, layerOutputs[layers.length - 1], i);
		}
		// The next for-loop is a late addition. It updates the biases of the last
		// layer based on the gradients of the last layer.
		for (int i = 0; i < layers[layers.length - 1].numNodes; i++) {
			neuralNet.changeBiasBy(layers.length - 1, i, -learningRate * nextLayerGrads[i] * layers[layers.length - 1].activation.derivative(layerOutputs[layers.length - 1][i]));
		}
		for (int layerCt = layers.length - 2; layerCt >= 0; layerCt--) {
			for (int i = 0; i < layers[layerCt].numNodes; i++) {
				for (int j = 0; j < layers[layerCt + 1].numNodes; j++) {
					double dw = -learningRate * nextLayerGrads[j] * layers[layerCt + 1].activation.derivative(layerOutputs[layerCt + 1][j])
							* (layerCt == 0 ? input[i] : layers[layerCt].activation.activationFunction(layerOutputs[layerCt][i]));
					neuralNet.changeWeightBy(layerCt, i, j, dw);
				}
			}
			double[] newNextLayerGrads = new double[layers[layerCt].numNodes];
			for (int i = 0; i < newNextLayerGrads.length; i++) {
				for (int j = 0; j < nextLayerGrads.length; j++) {
					newNextLayerGrads[i] += nextLayerGrads[j] * neuralNet.getWeight(layerCt, i, j) * layers[layerCt + 1].activation.derivative(layerOutputs[layerCt + 1][j]);
				}
			}
			nextLayerGrads = newNextLayerGrads;
			// As of this codepoint, nextLayerGrads are actually currentLayerGrads for this
			// iteration
			// as they are meant to be used in the next iteration. So:
			// Bias gradients actually depend on gradients of the layer they are at, not the
			// next layer. So using nextLayerGrads here actually has the functionality of
			// using the same layer grads for the biases, as the "nextLayerGrads" is
			// supposed to be used for the next iteration of layerCt. By using it right
			// here,... well you get the gist.
			// Note from future self: The above comment was meant to be comprehensible?
			// Note from even more future self: Read it properly, it is perfect.
			if (layerCt != 0) {
				for (int i = 0; i < layers[layerCt].numNodes; i++) {
					neuralNet.changeBiasBy(layerCt, i, -learningRate * nextLayerGrads[i] * layers[layerCt].activation.derivative(layerOutputs[layerCt][i]));
				}
			}
			// Note to self: Never hurry in life; wrote activation.activationFunction()
			// instead of activation.derivative() in previous line. Result: 10-mins of
			// hair-tearing.
		}
	}

	public void applySGD() {
		traindataset.shuffle();
		for (int i = 0; i < traindataset.records.length; i++) {
			trainSample(i);
		}

	}

	private void trainSample(int i) {
		Record rec = traindataset.records[i];
		trainSample(rec.input, rec.output);
	}

	public double getError() {
		double error = 0;
		for (int k = 0; k < traindataset.records.length; k++) {
			double[] target = traindataset.records[k].output;
			double[] output = this.neuralNet.predict(false, traindataset.records[k].input);
			error += this.errorfunc.getError(target, output);
		}
		error /= traindataset.records.length;
		return error;

	}

	/**
	 * Better use the other fit method
	 *
	 * @deprecated
	 */
	@Deprecated
	public void fit(DataSet dataset) throws InterruptedException {
		this.traindataset = dataset.applyConversionRule(neuralNet.inputConversionRule);
		int t1 = 0;
		while (true) {
			t1++;
			applySGD();
			perEpoch(t1, getError());
		}
	}

	public void fit(DataSet dataset, int maxIteration) throws InterruptedException {
		this.traindataset = dataset.applyConversionRule(neuralNet.inputConversionRule);
		int t1 = 0;
		while (t1 < maxIteration) {
			t1++;
			applySGD();
			perEpoch(t1, getError());
		}
	}

	public void perEpoch(int time, double error) throws InterruptedException {
		System.out.println(time + " " + error + " " + learningRate);
	}
}