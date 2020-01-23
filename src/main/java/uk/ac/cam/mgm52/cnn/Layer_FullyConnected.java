package uk.ac.cam.mgm52.cnn;

public class Layer_FullyConnected implements Layer {

    //We can think of weights as a series of input-sized weight tensors each associated with an output value
    Tensor weights;
    double[] biases;

    int[] expectedInputSizes;
    int[] outputDims;

    double[] recentOutput;
    Tensor recentInput;

    public Layer_FullyConnected(int outputLength, int[] inputDimSizes){
        expectedInputSizes = inputDimSizes;
        recentInput = new Tensor(inputDimSizes);
        recentOutput = new double[outputLength];
        outputDims = new int[] {outputLength};

        weights = new Tensor(ArrayUtils.appendValue(inputDimSizes, outputLength));

        //Following the initialization strategy proposed by He et.  2015 https://arxiv.org/pdf/1502.01852.pdf
        double randLimits = Math.sqrt(2) / Math.sqrt(ArrayUtils.product(inputDimSizes));
        weights = weights.randomsSND().product(randLimits);

        biases = new double[outputLength];
    }

    @Override
    public Tensor forwardProp(Tensor input) {
        recentInput = input;
        Tensor.RegionsIterator i = weights.new RegionsIterator(expectedInputSizes, new int[0]);

        while(i.hasNext()){
            recentOutput[i.coordIterator.getCurrentCount()] = i.next().innerProduct(input);
        }

        recentOutput = ArrayUtils.addAll(recentOutput, biases);
        return new Tensor(new int[] {recentOutput.length}, recentOutput);
    }


    //TODO: consider creating an easier way of doing some process to a tensor over an iterating region
    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {
        //outputGrad = derivative of loss wrt bias output

        //derivative of loss wrt weights = input * deriv wrt bias output
        Tensor weightGrad = null;
        for(int i = 0; i < outputGrad.values.length; i++){
            Tensor newGrads = recentInput.product(outputGrad.values[i]);

            weightGrad = (weightGrad == null) ? newGrads : weightGrad.appendTensor(newGrads, weights.dimSizes.length);
        }

        //derivative of loss wrt inputs = weights * deriv wrt bias output
        Tensor inputGrad = new Tensor(expectedInputSizes);
        Tensor.RegionsIterator j = weights.new RegionsIterator(expectedInputSizes, new int[0]);
        while(j.hasNext()){
            Tensor newGrads = j.next().product(outputGrad.values[j.coordIterator.getCurrentCount()-1]);

            inputGrad = inputGrad.add(newGrads, 1);
        }

        //Adjusting weights & biases
        weights = weights.add(weightGrad, -1.0 * learningRate);
        biases = ArrayUtils.addAll(biases, ArrayUtils.multAll(outputGrad.values, -1.0 * learningRate));

        return inputGrad;
    }

    @Override
    public int[] getOutputDims() {
        return outputDims;
    }
}
