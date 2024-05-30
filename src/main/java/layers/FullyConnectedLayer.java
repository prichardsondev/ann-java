package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {

    private double[][] _weights;
    private int _inLength;
    private int _outLength;
    private long _seed;
    private final double leak = 0.01;

    private double _learningRate;
    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int inLength, int outLength, long seed, double learningRate) {
        _inLength = inLength;
        _outLength = outLength;
        _seed = seed;
        _weights = new double[inLength][outLength];
        _learningRate = learningRate;

        setRandomWeights();
    }

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

    @Override
    //how much each weight contributed to error
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


    public void setRandomWeights(){
        Random r = new Random(_seed);

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = r.nextGaussian();
            }
        }
    }

    //activation function
    public double relu(double input){
        return input <= 0 ? 0 : input;
    }

    public double sigmoid(double input){
        return 1/(1+Math.exp(-input));
    }

    public double derivativeSigmoid(double input){
        return input*(1-input);
    }

    public double derivativeReLu(double input){
        return input <= 0 ? leak : 1;
    }
}
