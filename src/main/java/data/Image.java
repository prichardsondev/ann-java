package data;

public class Image {

    private final double [][] data;
    private final int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;

    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        String s = label + "\n";

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                s += data[i][j] + ",";
            }
            s += "\n";
        }
        return s;
    }
}
