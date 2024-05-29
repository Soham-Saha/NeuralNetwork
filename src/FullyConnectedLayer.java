import java.io.Serializable;

public class FullyConnectedLayer implements Serializable {
	Layer layerIn;
	Layer layerOut;
	double[][] edgeWeights;

	public FullyConnectedLayer(Layer layerIn, Layer layerOut) {
		this.layerIn = layerIn;
		this.layerOut = layerOut;
		this.edgeWeights = new double[layerIn.numNodes][layerOut.numNodes];
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		FullyConnectedLayer fcl = new FullyConnectedLayer(layerIn, layerOut);
		fcl.layerIn = (Layer) layerIn.clone();
		fcl.layerOut = (Layer) layerOut.clone();
		fcl.edgeWeights = edgeWeights.clone();
		return fcl;
	}
}
