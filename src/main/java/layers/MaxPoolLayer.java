package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {

    private final int _stepSize;
    private final int  _windowSize;

    private final int _inLength;
    private final int _inRows;
    private final int _inCols;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxCol;


    public MaxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inCols) {
        _stepSize = stepSize;
        _windowSize = windowSize;
        _inLength = inLength;
        _inRows = inRows;
        _inCols = inCols;
    }

    public List<double[][]> forwardPass(List<double[][]> input){
        List<double[][]> output = new ArrayList<>();

        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }

        return output;

    }

    private double[][] pool(double[][] input) {
        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for(int r = 0; r < getOutputRows(); r+=_stepSize){
            for(int c = 0; c < getOutputCols(); c+=_stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for(int x = 0; x < _windowSize; x++){
                    for(int y = 0; y < _windowSize; y++){
                        if(max < input[r+x][c+y]){
                            max = input[r+x][c+y];
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }
                    }
                }

                output[r][c] = max;

            }

        }

        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = forwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inLength,_inRows,_inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagate(List<double[][]> dLdO) {
        List<double[][]> dxdl = new ArrayList<>();

        int l = 0;
        for(double[][] array : dLdO){
            double[][] error = new double[_inRows][_inCols];

            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if (max_i != -1) error[max_i][max_j] += array[r][c];
                }
            }

            dxdl.add(error);
            l++;
        }

        if (_previousLayer!=null)_previousLayer.backPropagate(dxdl);
    }

    @Override
    public void backPropagate(double[] dLdO) {
        List<double[][]> matrixList =
                vectorToMatrix(dLdO,getOutputLength(),getOutputRows(),getOutputCols());

        backPropagate(matrixList);
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength * getOutputRows() * getOutputCols();
    }
}
