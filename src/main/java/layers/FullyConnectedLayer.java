package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {

    private final double[][] _weights;
    private final int _inLength;
    private final int _outLength;
    private final long _seed;
    private final double leak = 0.01;

    private final double _learningRate;
    private double[] lastZ;
    private double[] lastX;

    /**
     * Constructs a fully connected layer.
     *
     * @param inLength      Length of the input.
     * @param outLength     Length of the output.
     * @param seed          Seed for random initialization.
     * @param learningRate  Learning rate for updating weights.
     */
    public FullyConnectedLayer(int inLength, int outLength, long seed, double learningRate) {
        _inLength = inLength;
        _outLength = outLength;
        _seed = seed;
        _weights = new double[inLength][outLength];
        _learningRate = learningRate;

        setRandomWeights();
    }

    /**
     * Performs the forward pass through the fully connected layer.
     *
     * @param input Input data.
     * @return Output of the layer.
     */
    public double[] forwardPass(double[] input){
        lastX = input;
        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                z[j] += input[i]*_weights[i][j];
            }
        }

        lastZ = z;

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                out[j] = relu(z[j]);
            }
        }

        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = forwardPass(input);

        if(_nextLayer != null)
            return _nextLayer.getOutput(forwardPass);
        else return forwardPass;
    }

    /**
     * Performs the backward pass through the layer.
     * During the backward pass, the gradient of the loss function with respect to the layer output
     * (often denoted as dL/dO) is propagated backward through the layer to calculate the gradients
     * with respect to the layer parameters (weights and biases) and the inputs to the layer.
     *
     * This process involves using the chain rule of calculus to compute the gradients of the loss
     * function with respect to the layer parameters and inputs.
     *
     * @param dLdO  Gradient of the loss function with respect to the layer output.
     *              This gradient indicates how much the loss function would increase or decrease
     *              if the layer output were to increase by a small amount.
     */
    @Override
    public void backPropagate(double[] dLdO) {
        /*
            derivative function - slope of tangent line of a point on original function
            dy/dx or f'(x) = derivative notation

            f(x) = x^4 - 2x^3 - x^2 + 4x -1
            f'(x) = 4x^3 + 2*3x^2 - 2x + 4
            f'(x) = 4x^2+6x^2-2x+4

            chain rule - outside inside rule
            dy/dx = (f'(x)o * i ) * (f'(x)i)

            y = (3x+1)^7
            dy/dx = (7(3x+1)^6) * ( 3 )
                  = 21(3x+1)^6

            differentiate with respect to
            f(x,y)
            f'(x,y) with respect to x = df/dx
            f'(x,y) with respect to y = df/dy

        */
        double[] dLdX = new double[_inLength];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for(int k = 0; k < _inLength; k++){

            double dLdX_sum = 0;

            for(int j = 0; j < _outLength; j++){

                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = _weights[k][j];

                dLdw = dLdO[j]*dOdz*dzdw;

                _weights[k][j] -= dLdw*_learningRate;

                dLdX_sum += dLdO[j]*dOdz*dzdx;
            }

            dLdX[k] = dLdX_sum;
        }

        if(_previousLayer!= null) _previousLayer.backPropagate(dLdX);

    }

    @Override
    public void backPropagate(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagate(vector);
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    /**
     * Sets random weights for the fully connected layer.
     */
    public void setRandomWeights(){
        Random r = new Random(_seed);

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = r.nextGaussian();
            }
        }
    }

    /**
     * Rectified Linear Unit (ReLU) activation function.
     * ReLU introduces non-linearity to the output by thresholding the input at zero.
     *
     * @param input Input value.
     * @return Output after applying ReLU.
     */
    public double relu(double input){
        return input <= 0 ? 0 : input;
    }

    /**
     * Derivative of the ReLU activation function.
     *
     * @param input Input value.
     * @return Derivative value.
     */
    public double derivativeReLu(double input){
        return input <= 0 ? leak : 1;
    }
}
