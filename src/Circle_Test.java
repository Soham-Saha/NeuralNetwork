import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * @author Soham Saha
 */

public class Circle_Test {
	static double lr = 1e-6;

	static Sequential nn;
	static int numRecords = 1000;
	static DataSet dataset;
	static Trainer trainer;
	static Sequential nnNow;
	static Animation anim;
	static int every = 100;

	private static void createNeuralNetwork() {
		nn = new Sequential(2, 16, 16, 2) {
			@Override
			public void initializeBiases() {
				for (Layer l : layers) {
					for (int i = 0; i < l.numNodes; i++) {
						l.biases[i] = -Math.random() * 10;
					}
				}
			}
		};
		nn.inputConversionRule = new ConversionRule() {

			@Override
			public double[] run(double... array) {
				return new double[] { array[0] - 100, array[1] - 100 };
			}
		};
		/*-nn.inputConversionRule = new ConversionRule() {
		
			@Override
			public double[] run(double... array) {
				return new double[] { Math.sqrt((array[0] - 100) * (array[0] - 100) + (array[1] - 100) * (array[1] - 100)), Math.atan2(array[1] - 100, array[0] - 100) };
			}
		};*/

	}

	private static void configureTrainer() {
		trainer = new Trainer(nn, ErrorFunctions.getErrorFunction(ErrorFunctions.SOFTMAX_CROSSENTROPY), lr) {
			@Override
			public void perEpoch(int time, double error) throws InterruptedException {
				if (time % every == 0) {
					super.perEpoch(time, error);
					anim.drawNow();
				}
			}
		};
	}

	@SuppressWarnings("serial")
	private static void miscAndStart() throws InterruptedException {
		anim = new Animation("", new Dimension(200, 200), Color.black, 6) {

			@Override
			public void draw(Graphics2D g, int t) {
				for (int i = 0; i < 200; i++) {
					for (int j = 0; j < 200; j++) {
						double col = 255 * Sequential.softmax(nn.predict(new double[] { i, j }))[0];
						col = Math.floor(col / 50) * 50;
						g.setColor(new Color((int) col, (int) col, (int) col));
						g.fillRect(i, j, 1, 1);
					}
				}
				for (int i = 0; i < dataset.records.length; i++) {
					double col = dataset.records[i].output[0];
					if (col == 0) {
						g.setColor(Color.red);
					} else {
						g.setColor(Color.green);
					}
					g.fillRect((int) dataset.records[i].input[0], (int) dataset.records[i].input[1], 1, 1);
				}

			}
		};
		anim.display();
		anim.drawNow();
		trainer.fit(dataset);
	}

	private static void createDataset() throws IOException {
		dataset = new DataSetGenerator(numRecords) {

			BufferedImage img = ImageIO.read(new File("C:\\Users\\SANTANU\\Desktop\\target.png"));

			@Override
			protected Record createRecord(int i) {
				double[] input = new double[] { Math.random() * 200, Math.random() * 200 };
				double[] target = generateTarget(input);
				return new Record(input, target);
			}

			double[] generateTarget(double[] input) {
				if (img.getRGB((int) input[0], (int) input[1]) != Color.white.getRGB()) {
					return new double[] { 1, 0 };
				} else {
					return new double[] { 0, 1 };
				}
				/*-if ((input[0] - 100) * (input[0] - 100) + (input[1] - 100) * (input[1] - 100) < 2500) {
				return new double[] { 1, 0 };
				} else {
				return new double[] { 0, 1 };
				}*/
			}
		}.generateDataSet();
	}

	public static void main(String[] args) throws InterruptedException, IOException {
		createDataset();
		createNeuralNetwork();
		configureTrainer();
		miscAndStart();
	}

}