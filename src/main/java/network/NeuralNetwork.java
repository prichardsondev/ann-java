package network;

import data.Image;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class NeuralNetwork {

    List<Layer> _layers;
    double _scaleFactor;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
       _layers = layers;
       _scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers() {
        if(_layers.size() <= 1) return;
        //removed -1
        for (int i = 0; i < _layers.size(); i++) {
            if(i==0)
                _layers.get(i).set_nextLayer(_layers.get(i + 1));
            else if (i == _layers.size()-1)
                _layers.get(i).set_previousLayer(_layers.get(i - 1));
            else {
                _layers.get(i).set_previousLayer(_layers.get(i - 1));
                _layers.get(i).set_nextLayer(_layers.get(i + 1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];

        expected[correctAnswer] = 1;

        return add(networkOutput, multiply(expected, -1));
    }

    private int getMaxIndex(double[] in){

        double max = in[0];
        int maxIndex = 0;

        for (int i = 1; i < in.length; i++) {
            if(in[i] > max){
                max = in[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public int guess(Image image) {
        List<double[][]>  inList = new ArrayList<>();
        inList.add(multiply(image.getData(), (1/_scaleFactor)));
        double[] output = _layers.getFirst().getOutput(inList);
        int maxIndex = getMaxIndex(output);
        return getMaxIndex(output);
    }

    public float test (List<Image> images) {
        int correct = 0;
        for (Image image : images) {
            int guess = guess(image);
            if(guess == image.getLabel())
                correct++;
        }
        return (float)correct/images.size();
    }

    public void train(List<Image> images) {
        for (Image image : images) {

            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(image.getData(), (1/_scaleFactor)));

            double[] output = _layers.getFirst().getOutput(inList);

            double[] dLdO = getErrors(output, image.getLabel());

            _layers.getLast().backPropagate(dLdO);
        }
    }


}
