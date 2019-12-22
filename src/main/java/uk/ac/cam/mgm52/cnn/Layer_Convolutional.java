package uk.ac.cam.mgm52.cnn;

public class Layer_Convolutional implements Layer {

    Tensor filters;
    int[] filterDimSizes;

    Tensor recentInput;

    /**Set up layer.
     * @param filterDimSizes Size of each filter
     * @param depth Number of filters
     */
    public Layer_Convolutional(int[] filterDimSizes, int depth){
        //Set up a single tensor that represents all filters
        this.filterDimSizes = filterDimSizes;
        filters = new Tensor(ArrayUtils.appendValue(filterDimSizes, depth));

        //Initialize values to randoms, varying with size of each filter.
        filters = filters.randoms(0.0, 1.0/(filters.getValuesCopy().length / depth));
    }

    @Override
    public Tensor forwardProp(Tensor input) {

        recentInput = input;
        Tensor output = null;

        //Iterate through each filter, appending result to output tensor.
        for (Tensor.TensorRegionsIterator i = filters.new TensorRegionsIterator(filterDimSizes); i.hasNext(); ) {
            Tensor newMap = input.crossCorrelationMap(i.next());

            //Append or assign new map to output
            output = (output == null) ? newMap : output.appendTensor(newMap);
        }

        return output;
    }

    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {

        //Initialise output to null. Easier than calculating size of first result.
        Tensor filterGrads = null;

        Tensor newFilters = null;

        //We can think of outputGrad as a series of "gradient filters" which we apply to the recent input.
        //This is the size of each of those filters.
        int[] outputGradSize = outputGrad.getFirstDimsCopy(outputGrad.rank - 1);

        //Iterate through each output grad, appending result to filter grad tensor.
        //Simultaneously, iterate through our original filters, altering them via gradient descent
        Tensor.TensorRegionsIterator i = outputGrad.new TensorRegionsIterator(outputGradSize);
        Tensor.TensorRegionsIterator j = filters.new TensorRegionsIterator(filterDimSizes);

        while ( i.hasNext() && j.hasNext()) {
            Tensor newMap = recentInput.crossCorrelationMap(i.next());

            //Append or assign new map to output
            filterGrads = (filterGrads == null) ? newMap : filterGrads.appendTensor(newMap);

            //Gradient descent on each filter
            Tensor newFilter = j.next().add(newMap, -1 * learningRate);

            newFilters = (newFilters == null) ? newFilter : newFilters.appendTensor(newFilter);
        }

        filters = newFilters;

        return filterGrads;
    }
}
