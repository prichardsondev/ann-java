package data;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    private static final int rows = 28;
    private static final int cols = 28;

    public static List<Image> readData(String path){
        List<Image> images = new ArrayList<Image>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;

            while ((line = br.readLine()) != null) {
                String[] lineData = line.split(",");

                double[][] data = new double[rows][cols];
                int label = Integer.parseInt(lineData[0]);
                int i = 1;

                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        data[row][col] = Double.parseDouble(lineData[i]);
                        i++;
                    }
                }

                images.add(new Image(data, label));
            }
        } catch (Exception e){}

        return images;
    }
}
