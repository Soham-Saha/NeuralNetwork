import java.io.Serializable;

public class DataSet implements Serializable {
	Record[] records;

	public DataSet(Record... records) {
		this.records = records;
	}

	public DataSet(int numRecords) {
		this.records = new Record[numRecords];
		for (int i = 0; i < numRecords; i++) {
			this.records[i] = new Record(null, null);
		}
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		Record[] newrec = records.clone();
		return new DataSet(newrec);
	}

	public DataSet applyConversionRule(ConversionRule inputConversionRule) {
		int num = this.records.length;
		DataSet out = new DataSet(num);
		for (int i = 0; i < num; i++) {
			out.records[i] = new Record(inputConversionRule.run(this.records[i].input), this.records[i].output);
		}
		return out;
	}

	public void shuffle() {
		for (int i = records.length - 1; i > 0; i--) {
			int j = (int) (Math.random() * records.length);
			Record temp = records[i];
			records[i] = records[j];
			records[j] = temp;
		}
	}
}
