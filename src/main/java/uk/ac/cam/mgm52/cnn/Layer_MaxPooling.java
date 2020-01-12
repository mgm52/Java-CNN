package uk.ac.cam.mgm52.cnn;

import java.util.ArrayList;

public class Layer_MaxPooling implements Layer{

    int[] strides;
    int[] sizes;

    Tensor outputTensor;

    Tensor inputTensor;

    //indices of each max value, used in backprop later
    int[] maxIndices;

    public Layer_MaxPooling(int[] strides, int[] sizes, int[] expectedInputDims){
        while(strides.length < expectedInputDims.length) strides = ArrayUtils.appendValue(strides, 1);
        this.strides = strides;

        while(sizes.length < expectedInputDims.length) sizes = ArrayUtils.appendValue(sizes, 1);
        this.sizes = sizes;

        inputTensor = new Tensor(expectedInputDims);

        int[] outputDims = new int[expectedInputDims.length];
        for(int i = 0; i < outputDims.length; i ++){
                outputDims[i] = (int) Math.ceil((expectedInputDims[i] - sizes[i] + 1)/(double)strides[i]);
        }
        outputTensor = new Tensor(outputDims);

        maxIndices = new int[outputTensor.values.length];
    }

    @Override
    public Tensor forwardProp(Tensor input) {
        //Apply max function across input
        for (Tensor.RegionsIterator i = input.new RegionsIterator(sizes, new int[0], strides); i.hasNext(); ) {
            Tensor nextRegion = i.next();
            int maxIndex = nextRegion.maxValueIndex();

            maxIndices[i.coordIterator.getCurrentCount()-1] = maxIndex;
            outputTensor.values[i.coordIterator.getCurrentCount()-1] = nextRegion.values[maxIndex];
        }

        return outputTensor;
    }


    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {
        Tensor inputGrads = inputTensor.zeroes();

        //In this loop, gradients are paired with the element that previously returned a max value
        for (Tensor.RegionsIterator i = inputGrads.new RegionsIterator(sizes, new int[0], strides); i.hasNext(); ) {
            i.next();

            int maxIndex = maxIndices[i.coordIterator.getCurrentCount()-1];

            //Coords of max, relative to region start
            int[] maxCoords = HornerConversion.hornerToCoords(maxIndex, sizes);

            //Start of region within base input tensor
            int[] regionStart = i.coordIterator.getCurrentCoords();

            //Coords of max, relative to base input tensor start
            int[] coordsOfMax = ArrayUtils.addAll(regionStart, maxCoords);

            inputGrads.set(coordsOfMax, outputGrad.values[i.coordIterator.getCurrentCount() - 1]);
        }

        return inputGrads;
    }

    @Override
    public int[] getOutputDims() {
        return outputTensor.dimSizes;
    }
}
