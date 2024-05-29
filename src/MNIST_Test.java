import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import javax.imageio.ImageIO;

public class MNIST_Test {
	static DataSet ds;
	static DataSet test;
	static Sequential nn;
	static boolean[][] data = new boolean[28][28];

	public static void main(String[] args) throws InterruptedException, FileNotFoundException, IOException, ClassNotFoundException {
		predictor();
	}

	private static void predictor() throws FileNotFoundException, IOException, ClassNotFoundException, InterruptedException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream("mnist_dontrewrite.txt"));
		nn = (Sequential) in.readObject();
		in.close();
		Animation anim = new Animation("Predictor", new Dimension(280 + 150, 280), Color.black, 16) {
			double[] output = new double[10];

			@Override
			public void updateData(int t) {
				int k = 0;
				int xMean = 0;
				int yMean = 0;
				for (int i = 0; i < 28; i++) {
					for (int j = 0; j < 28; j++) {
						if (data[i][j]) {
							xMean += i - 14;
							yMean += j - 14;
							k++;
						}
					}
				}
				if (k == 0) {
					output = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				} else {
					double[] input = new double[784];
					xMean /= k;
					yMean /= k;
					k = 0;
					for (int i = 0; i < 28; i++) {
						for (int j = 0; j < 28; j++) {
							try {
								input[k] = data[i + xMean][j + yMean] ? 1 : 0;
							} catch (Exception e) {
								// TODO: handle exception
							}
							k++;
						}
					}
					output = Sequential.softmax(nn.predict(input));
				}
			}

			@Override
			public void draw(Graphics2D g, int t) {
				g.setColor(Color.white);
				for (int x = 0; x < 28; x++) {
					for (int y = 0; y < 28; y++) {
						if (data[x][y]) {
							g.fillRect(10 * x, 10 * y, 11, 11);
						}
					}
				}
				g.drawLine(280, 0, 280, 280);
				// double n = 1.5;
				// g.drawRect((int) (28 * n), (int) (28 * n), (int) (280 - 2 * 28 * n), (int)
				// (280 - 2 * 28 * n));
				for (int i = 0; i < 10; i++) {
					g.drawString(i + "", 300, 16 * (i + 1) + 10);
					g.fillRect(320, 16 * i + 16, (int) (100 * output[i]), 10);
				}

			}

		};
		anim.addMouseMotionListener(new MouseMotionAdapter() {

			@Override
			public void mouseDragged(MouseEvent e) {
				try {
					data[e.getX() / 10][e.getY() / 10 - 2] = true;
				} catch (Exception e2) {
					// TODO: handle exception
				}
			}
		});
		anim.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				for (int i = 0; i < 28; i++) {
					for (int j = 0; j < 28; j++) {
						data[i][j] = false;
					}
				}

			}
		});
		anim.display();
		double[] input = new double[784];
		while (true) {
			Thread.sleep(1000 / 16);
			anim.drawNow();
		}
	}

	private static void trainer() throws FileNotFoundException, IOException, ClassNotFoundException, InterruptedException {
		// createTestSet(500);
		// createTrainSet(1000);
		// nn = new Sequential(784, 10);
		ObjectInputStream in = new ObjectInputStream(new FileInputStream("mnist_empty_1.txt"));
		nn = (Sequential) in.readObject();
		in.close();
		readOldTestSet();
		readOldTrainSet();
		Trainer tr = new Trainer(nn, ErrorFunctions.SOFTMAX_CROSSENTROPY, 1e-6) {
			@Override
			public void perEpoch(int time, double error) throws InterruptedException {
				// super.perEpoch(time, error);
				System.out.println(time + "\t" + MNIST_Test.testAccuracy());
				try {
					System.out.println("Writing");
					ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("mnist_empty_1.txt"));
					out.writeObject(nn);
					out.close();
					System.out.println("Written");
				} catch (Exception e) {
					// TODO: handle exception
				}
			}
		};
		tr.fit(ds);

	}

	private static void readOldTrainSet() throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream("trainset.txt"));
		ds = (DataSet) in.readObject();
		in.close();

	}

	private static void readOldTestSet() throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream("testset.txt"));
		test = (DataSet) in.readObject();
		in.close();

	}

	private static void createTestSet(int n) {
		test = new DataSet(n * 10);
		int i = 0;
		for (File x : new File("C:\\Users\\SANTANU\\Downloads\\MNIST Dataset JPG format\\MNIST Dataset JPG format\\MNIST - JPG - testing").listFiles()) {
			int ct = 0;
			for (File f : x.listFiles()) {
				test.records[i] = new Record(encode(f), encodeOutput(x));
				i++;
				ct++;
				if (ct >= n) {
					break;
				}
			}
		}
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("testset.txt"));
			out.writeObject(test);
			out.close();
			System.out.println("Written.");
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

	private static void createTrainSet(int n) {
		ds = new DataSet(n * 10);
		int i = 0;
		for (File x : new File("C:\\Users\\SANTANU\\Downloads\\MNIST Dataset JPG format\\MNIST Dataset JPG format\\MNIST - JPG - training").listFiles()) {
			int ct = 0;
			for (File f : x.listFiles()) {
				ds.records[i] = new Record(encode(f), encodeOutput(x));
				i++;
				ct++;
				if (ct >= n) {
					break;
				}
			}
		}
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("trainset.txt"));
			out.writeObject(ds);
			out.close();
			System.out.println("Written.");
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

	private static double[] encodeOutput(File x) {
		double out[] = new double[10];
		out[Integer.parseInt(x.getName())] = 1;
		return out;
	}

	public static double[] encode(File f) {
		BufferedImage img = null;
		try {
			img = ImageIO.read(f);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		double[] output = new double[784];
		int i = 0;
		for (int x = 0; x < 28; x++) {
			for (int y = 0; y < 28; y++) {
				output[i] = img.getRGB(x, y) != Color.black.getRGB() ? 1 : 0;
				i++;
			}
		}
		return output;
	}

	public static double testAccuracy() {
		double good = 0;
		for (Record r : test.records) {
			double[] output = Sequential.softmax(nn.predict(r.input));
			int k = 0;
			double max = 0;
			for (int i = 0; i < output.length; i++) {
				if (output[i] > max) {
					max = output[i];
					k = i;
				}
			}
			if (r.output[k] == 1) {
				good++;
			}
		}
		return good / test.records.length;

	}

}