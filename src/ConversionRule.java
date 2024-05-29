import java.io.Serializable;

public interface ConversionRule extends Serializable {
	double[] run(double... array);
}
