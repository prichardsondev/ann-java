package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class ConvolutionLayer extends Layer {

    private long _seed;

    private List<double[][]> _filters;
    private int _filterSize;
    private int _stepSize;
    private int _numFilters;
    private int _inLength;
    private int _inRows;
    private int _inCols;
    private double _learningRate;

    private List<double[][]> _lastInput;

    public ConvolutionLayer(int filterSize, int stepSize, int inLength, int inRows,
                            int inCols, long seed, int numFilters, double learningRate) {
        _filterSize = filterSize;
        _stepSize = stepSize;
        _inLength = inLength;
        _inRows = inRows;
        _inCols = inCols;
        _seed = seed;
        _numFilters = numFilters;
        _learningRate = learningRate;

        generateRandomFilters(numFilters);
    }

    private void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(_seed);
        for (int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[_filterSize][_filterSize];

            for (int i = 0; i < _filterSize; i++) {
                for (int j = 0; j < _filterSize; j++) {
                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);

        }

        _filters = filters;

    }

    public List<double[][]> forwardPass(List<double[][]> list) {

        _lastInput = list;

        List<double[][]> output = new ArrayList<>();

        for (int m = 0; m < list.size(); m++) {
            for (double[][] filter : _filters){
                output.add(convolve(list.get(m), filter,_stepSize));
            }
        }
        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {

        int outRows = (input.length - filter.length)/stepSize+1;
        int outCols = (input[0].length - filter[0].length)/stepSize+1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol = 0;

        for (int i = 0; i <= inRows - fRows; i+=stepSize) {
            outCol = 0;
            for (int j = 0; j <= inCols - fCols; j+=stepSize) {
                double sum = 0;
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                       int inputRowIndex = i+x;
                       int inputColIndex = j+y;

                       sum+= filter[x][y] * input[inputRowIndex][inputColIndex];
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;

            }

            outRow++;
        }

        return  output;
    }

    public double[][] spaceArray(double[][] input){
        if(_stepSize ==1 ) return input;

        int outRows = (input.length - 1)*_stepSize + 1;
        int outCols = (input[0].length - 1)*_stepSize + 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i*_stepSize][j*_stepSize] = input[i][j];
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = forwardPass(input);

        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, _inLength,_inRows, _inCols);
        return getOutput(matrixInput);
    }

    @Override
    public void backPropagate(List<double[][]> dLdO) {
        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPrevious = new ArrayList<>();

        for (int f = 0; f < _filters.size(); f++) {
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        for (int i = 0; i < _lastInput.size(); i++) {
            double[][] errorForInput = new double[_inRows][_inCols];
            for (int f = 0; f < _filters.size(); f++) {
                double[][] currentFilter = _filters.get(f);
                double[][] error = dLdO.get(i*_filterSize+f);

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(dLdF, _learningRate*-1);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontally(flipArrayVertically(spacedError));

                errorForInput =  add(errorForInput, fullConvolve(currentFilter, flippedError));
            }

            dLdOPrevious.add(errorForInput);
        }

        for (int f = 0; f < _filters.size(); f++) {
            double[][] modifiedFilter = add( filtersDelta.get(f),_filters.get(f));
            _filters.set(f,modifiedFilter);
        }

        if(_previousLayer != null){
            _previousLayer.backPropagate(dLdOPrevious);
        }
    }

    public void backPropagate(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, _inLength,_inRows, _inCols);
        backPropagate(matrixInput);
    }

    public double[][] flipArrayHorizontally(double[][] input){
        double[][] output = new double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i][j] = input[i][input[0].length-j-1];
            }
        }

        return output;
    }

    public double[][] flipArrayVertically(double[][] input){
        double[][] output = new double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i][j] = input[input.length-i-1][j];
            }
        }

        return output;
    }

    public double[][] flipArrayDiagonally(double[][] input){
        double[][] output = new double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i][j] = input[input.length-i-1][input[0].length-j-1];
            }
        }

        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {

        int outRows = (input.length + filter.length)+1;
        int outCols = (input[0].length + filter[0].length);

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol = 0;

        for (int i = -fRows + 1; i < inRows; i++) {
            outCol = 0;
            for (int j = -fCols + 1; j < inCols; j++) {
                double sum = 0;
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        if(inputRowIndex >= 0 && inputColIndex >=0 && inputRowIndex < inRows && inputColIndex < inCols){
                            sum+= filter[x][y] * input[inputRowIndex][inputColIndex];
                        }


                    }
                }

                output[outRow][outCol] = sum;
                outCol++;

            }

            outRow++;
        }

        return  output;
    }

    @Override
    public int getOutputLength() {
        return _filters.size()*_inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols-_filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols()*getOutputRows()*getOutputLength();
    }
}