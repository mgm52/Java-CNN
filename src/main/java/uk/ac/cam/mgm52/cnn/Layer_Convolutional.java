package uk.ac.cam.mgm52.cnn;

public class Layer_Convolutional implements Layer {

    Tensor filters;
    int[] filterDimSizes;
    int[] ccMapSize;

    int[] outputDims;

    Tensor recentInput;

    /**Set up layer.
     * @param filterDimSizes Size of each filter
     * @param depth Number of filters
     */
    public Layer_Convolutional(int[] filterDimSizes, int depth, int[] inputDims){
        //Set up a single tensor that represents all filters
        while(filterDimSizes.length < inputDims.length) filterDimSizes = ArrayUtils.appendValue(filterDimSizes, 1);
        this.filterDimSizes = filterDimSizes;

        //Calculate the size of the cross correlation map resultant from applying a given filter
        ccMapSize = ArrayUtils.subtractAll(inputDims, ArrayUtils.addAll(filterDimSizes, -1));

        filters = new Tensor(ArrayUtils.appendValue(filterDimSizes, depth));

        //Following the initialization strategy proposed by He et. al. 2015 https://arxiv.org/pdf/1502.01852.pdf
        double randLimits = Math.sqrt(2) / Math.sqrt(ArrayUtils.product(inputDims));
        filters = filters.randomsSND().product(randLimits);

        outputDims = ArrayUtils.appendValue(ccMapSize, depth);
    }

    /**Generate cross-correlation map of a filter applied to a tensor.
     * @param ccMapSize The expected size of the output. This is taken as a parameter to avoid recomputing it at each iteration.
     * */
    public static Tensor crossCorrelationMap(Tensor base, Tensor filter, int[] ccMapSize, int[] padding){
        double[] ccMapValues = new double[ArrayUtils.product(ccMapSize)];

        for (Tensor.RegionsIterator i = base.new RegionsIterator(filter.dimSizes, padding); i.hasNext(); ) {
            Tensor region = i.next();
            ccMapValues[i.coordIterator.getCurrentCount()-1] = region.innerProduct(filter);
        }

        return new Tensor(ccMapSize, ccMapValues);
    }

    @Override
    public Tensor forwardProp(Tensor input) {

        recentInput = input;
        Tensor output = null;

        //Iterate through each filter, appending result to output tensor.
        for (Tensor.RegionsIterator i = filters.new RegionsIterator(filterDimSizes, new int[0]); i.hasNext(); ) {
            Tensor newMap = crossCorrelationMap(input, i.next(), ccMapSize, new int[0]);

            //Append or assign new map to output
            output = (output == null) ? newMap : output.appendTensor(newMap, filters.rank);
        }

        return output;
    }


    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {
        //Initialise output to null. Easier than calculating size of first result.
        Tensor filterGrads = null;
        Tensor inputGrads = recentInput.zeroes();

        //We can think of outputGrad as a series of "gradient filters" which we apply to the recent input.
        //This is the size of each of those filters.
        int[] outputGradSize = outputGrad.getFirstDimsCopy(outputGrad.rank - 1);

        //Iterate through each output grad, appending result to filter grad tensor.
        //Simultaneously, iterate through our original filters, altering them via gradient descent
        Tensor.RegionsIterator i = outputGrad.new RegionsIterator(outputGradSize, new int[0]);
        Tensor.RegionsIterator j = filters.new RegionsIterator(filterDimSizes, new int[0]);


        while ( i.hasNext() && j.hasNext()) {
            Tensor outputLayer = i.next();
            Tensor filterLayer = j.next();

            //Calculating deriv wrt filters
            Tensor newMap = crossCorrelationMap(recentInput, outputLayer, filterDimSizes, new int[0]);
            //Append or assign new map to filter grad tensor
            filterGrads = (filterGrads == null) ? newMap : filterGrads.appendTensor(newMap, outputGrad.rank);

            //Deriv wrt input = deriv wrt outputs * flipped filters
            //To return the correct sized tensor, this requires some padding - which happens to be (filter sizes - 1)
            int[] padding = ArrayUtils.addAll(filterLayer.dimSizes, -1);
            Tensor flippedFilter = filterLayer.flip();
            Tensor currentInputGrad = crossCorrelationMap(outputLayer, flippedFilter, inputGrads.dimSizes, padding);
            inputGrads = inputGrads.add(currentInputGrad, 1);
        }

        //Gradient descent on filters
        filters = filters.add(filterGrads, -1 * learningRate);

        return inputGrads;
    }

    @Override
    public int[] getOutputDims() {
        return outputDims;
    }
}
