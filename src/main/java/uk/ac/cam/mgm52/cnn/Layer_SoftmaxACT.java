package uk.ac.cam.mgm52.cnn;

public class Layer_SoftmaxACT implements Layer {

    int[] outputDims;
    Tensor recentSoftMaxInput;

    public Layer_SoftmaxACT(int[] inputDims){
        outputDims = inputDims;
    }

    //How much these inputs will affect the result of softmax output
    public Tensor derivatives(Tensor input){
        double[] output = new double[input.values.length];

        double expSum = 0;

        for(int i = 0; i < input.values.length; i++){
            output[i] = Math.exp(input.values[i]);
            expSum += output[i];
        }

        for(int i = 0; i < input.values.length; i++){
            output[i] *= (expSum - output[i]);
        }

        return new Tensor(input.dimSizes, ArrayUtils.divideAll(output, expSum * expSum));
    }

    //softMax(Si) = (e^Si)/sum(e^S)
    @Override
    public Tensor forwardProp(Tensor input) {
        recentSoftMaxInput = input;

        Tensor output = input.zeroes();
        double expSum = 0;

        for(int i = 0; i < input.values.length; i++){
            output.values[i] = Math.exp(input.values[i]);
            expSum += output.values[i];
        }

        output.values = ArrayUtils.divideAll(output.values, expSum);
        return output;
    }

    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {
        return outputGrad.product(derivatives(recentSoftMaxInput));
    }

    @Override
    public int[] getOutputDims() {
        return outputDims;
    }
}
