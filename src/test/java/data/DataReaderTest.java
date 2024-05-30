package data;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DataReaderTest {

    @org.junit.jupiter.api.Test
    void readData() {
        String path = "data/mnist_test.csv";
        List<Image> images = DataReader.readData(path);
        assertFalse(images.isEmpty());
    }
}