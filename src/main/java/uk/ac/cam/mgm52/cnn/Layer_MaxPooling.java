package uk.ac.cam.mgm52.cnn;

import java.util.ArrayList;
import java.util.Arrays;

public class Layer_MaxPooling implements Layer{

    int[] strides;
    int[] sizes;
    Tensor recentInput;
    Tensor recentOutput;

    public Layer_MaxPooling(int[] strides, int[] sizes){
        this.strides = strides;
        this.sizes = sizes;
    }

    @Override
    public Tensor forwardProp(Tensor input) {
        recentInput = input;
        int[] inputDims = input.getFirstDimsCopy(input.rank);

        int valuesLength = 1;
        int[] newDims = new int[input.rank];

        for(int i = 0; i < newDims.length; i ++){
            newDims[i] = (int) Math.ceil((inputDims[i] - sizes[i] + 1)/(double)strides[i]);
            valuesLength *= newDims[i];
        }

        double[] maxValues = new double[valuesLength];

        //Apply max function across input
        for (Tensor.TensorRegionsIterator i = input.new TensorRegionsIterator(sizes, strides); i.hasNext(); ) {

            Tensor nextRegion = i.next();

            maxValues[i.coordIterator.getCurrentCount()-1] = nextRegion.maxValue();
        }

        recentOutput = new Tensor(newDims, maxValues);
        return recentOutput;
    }


    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {

        double[] recentOutputValues = recentOutput.getValuesCopy();
        double[] inputGradValues = new double[recentInput.getValuesCopy().length];
        double[] outputGradValues = outputGrad.getValuesCopy();
        int[] inputDimSizes = recentInput.getFirstDimsCopy(recentInput.rank);

        for (Tensor.TensorRegionsIterator i = recentInput.new TensorRegionsIterator(sizes, strides); i.hasNext(); ) {


            double[] regionValues = i.next().getValuesCopy();

            for(int j = 0; j < regionValues.length; j ++){
                if(regionValues[j] == recentOutputValues[i.coordIterator.getCurrentCount() - 1]){
                    //Found a max value! Now copy over gradient into new tensor.
                    int[] regionStart = i.coordIterator.getCurrentCoords();
                    int[] coordsInRegion = HornerConversion.hornerToCoords(j, sizes);

                    int[] coordsOfMax = ArrayUtils.addAll(regionStart, coordsInRegion);

                    int hornerOfMax = HornerConversion.coordsToHorner(coordsOfMax, inputDimSizes);
                    inputGradValues[hornerOfMax] = outputGradValues[i.coordIterator.getCurrentCount() - 1];
                }
            }
        }

        return new Tensor(inputDimSizes, inputGradValues);
    }
}
