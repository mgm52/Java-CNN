package uk.ac.cam.mgm52.cnn;

public class Layer_ReluACT implements Layer {

    Tensor recentInput;

    int[] outputDims;

    public Layer_ReluACT(int[] inputDims){
        outputDims = inputDims;
    }

    //How much these inputs will affect the result of softmax output
    public Tensor derivatives(Tensor input){
        Tensor output = input.zeroes();

        for(int i = 0; i < input.values.length; i++){
            output.values[i] = input.values[i] > 0 ? 1 : 0;
        }

        return output;
    }

    //softMax(Si) = (e^Si)/sum(e^S)
    @Override
    public Tensor forwardProp(Tensor input) {
        recentInput = input;
        Tensor output = input.zeroes();

        for(int i = 0; i < input.values.length; i++){
            output.values[i] = Math.max(input.values[i], 0);
        }

        return output;
    }

    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {
        return outputGrad.product(derivatives(recentInput));
    }

    @Override
    public int[] getOutputDims() {
        return outputDims;
    }
}
