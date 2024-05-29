abstract class DataSetGenerator {
	int numRecords;

	public DataSetGenerator(int numRecords) {
		this.numRecords = numRecords;
	}

	public DataSet generateDataSet() {
		DataSet ds = new DataSet(numRecords);
		for (int i = 0; i < numRecords; i++) {
			ds.records[i] = createRecord(i);
		}
		return ds;
	}

	protected abstract Record createRecord(int i);
}
