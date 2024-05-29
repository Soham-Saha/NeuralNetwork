import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.io.Serializable;

import javax.swing.JFrame;
import javax.swing.JPanel;

abstract class Animation extends JFrame implements Serializable {

	JPanel pnl = new JPanel();
	boolean running = true;
	String thisName;
	Dimension dim = new Dimension();
	Color bg;
	double del = 150;
	boolean noDelay;
	static int t = 0;

	public Animation(String thisName, Dimension dim, Color bg, double fps) {
		this.thisName = thisName;
		this.dim = dim;
		this.bg = bg;
		this.del = 1000 / fps;
		this.setTitle(thisName);
		this.setResizable(false);
		this.setSize(dim.width + 17, dim.height + 40);
		this.setLocation(Toolkit.getDefaultToolkit().getScreenSize().width / 2 - this.getSize().width / 2, Toolkit.getDefaultToolkit().getScreenSize().height / 2 - this.getSize().height / 2);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public void display() throws InterruptedException {
		this.setVisible(true);
	}

	public void drawNow() throws InterruptedException {
		t++;
		updateData(t);
		this.remove(pnl);
		pnl = new JPanel() {
			@Override
			protected void paintComponent(Graphics g) {
				if (bg != null) {
					g.setColor(bg);
					g.fillRect(0, 0, this.getWidth(), this.getHeight());
				}
				draw((Graphics2D) g, t);
			}

		};
		this.add(pnl);
		this.invalidate();
		this.validate();
		this.repaint();
		Thread.sleep((int) del);
	}

	public abstract void draw(Graphics2D g, int t);

	public void updateData(int t) {
	};

}
