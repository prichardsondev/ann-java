
package layers;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract class representing a layer in a neural network.
 */
public abstract class Layer {

    /**
     * Gets the next layer in the neural network.
     *
     * @return The next layer.
     */
    public Layer get_nextLayer() {
        return _nextLayer;
    }

    /**
     * Sets the next layer in the neural network.
     *
     * @param _nextLayer The next layer to set.
     */
    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }

    /**
     * Gets the previous layer in the neural network.
     *
     * @return The previous layer.
     */
    public Layer get_previousLayer() {
        return _previousLayer;
    }

    /**
     * Sets the previous layer in the neural network.
     *
     * @param _previousLayer The previous layer to set.
     */
    public void set_previousLayer(Layer _previousLayer) {
        this._previousLayer = _previousLayer;
    }

    protected Layer _nextLayer;
    protected Layer _previousLayer;

    /**
     * Computes the output of the layer given a list of input matrices.
     *
     * @param input List of input matrices.
     * @return The output as an array of doubles.
     */
    public abstract double[] getOutput(List<double[][]> input);

    /**
     * Computes the output of the layer given a single input array.
     *
     * @param input Input array.
     * @return The output as an array of doubles.
     */
    public abstract double[] getOutput(double[] input);

    /**
     * Backpropagates the gradient through the layer given a list of gradients.
     *
     * @param dLdO List of gradients with respect to the layer output.
     */
    public abstract void backPropagate(List<double[][]> dLdO);

    /**
     * Backpropagates the gradient through the layer given a single gradient array.
     *
     * @param dLdO Gradient with respect to the layer output.
     */
    public abstract void backPropagate(double[] dLdO);

    /**
     * Gets the length of the output array.
     *
     * @return The length of the output array.
     */
    public abstract int getOutputLength();

    /**
     * Gets the number of rows in the output matrix.
     *
     * @return The number of rows in the output matrix.
     */
    public abstract int getOutputRows();

    /**
     * Gets the number of columns in the output matrix.
     *
     * @return The number of columns in the output matrix.
     */
    public abstract int getOutputCols();

    /**
     * Gets the total number of elements in the output matrix.
     *
     * @return The total number of elements in the output matrix.
     */
    public abstract int getOutputElements();

    /**
     * Converts a list of matrices to a single vector.
     *
     * @param input List of matrices to convert.
     * @return The concatenated vector.
     */
    public double[] matrixToVector(List<double[][]> input) {

        int length = input.size();
        int rows = input.getFirst().length;
        int cols = input.getFirst()[0].length;

        double[] vector = new double[length*rows*cols];

        int i = 0;
        for (int l = 0; l < length; l++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    vector[i] = input.get(l)[r][c];
                    i++;
                }
            }
        }

        return vector;
    }

    /**
     * Converts a vector to a list of matrices.
     *
     * @param input Vector to convert.
     * @param length Number of matrices.
     * @param rows Number of rows in each matrix.
     * @param cols Number of columns in each matrix.
     * @return List of matrices.
     */
    List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols){
        List<double[][]> out = new ArrayList<>();

        int i = 0;
        for(int l = 0; l < length; l++ ){

            double[][] matrix = new double[rows][cols];

            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    matrix[r][c] = input[i];
                    i++;
                }
            }

            out.add(matrix);
        }

        return out;
    }

}

//https://www.youtube.com/watch?v=JJUlkPFq1q8&list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN&index=4
