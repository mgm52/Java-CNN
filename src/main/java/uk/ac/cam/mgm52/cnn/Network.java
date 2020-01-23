package uk.ac.cam.mgm52.cnn;

import java.util.Arrays;

/**An array of layers, connected to each other.*/
public class Network implements Layer {

    Layer[] layers = {};

    //This is used to return output dims in instances where no layers have been added yet
    private int[] expectedInputDims;

    public Network(Layer[] layers){
        this.layers = layers;
    }

    public Network(int... expectedInputDims){
        this.expectedInputDims = expectedInputDims;
    }


    public Network addConv(int filterSize, int filterCount){
        int[] filterDims = new int[getOutputDims().length];
        Arrays.fill(filterDims, filterSize);

        layers = ArrayUtils.appendValue(layers, new Layer_Convolutional(filterDims, filterCount, getOutputDims()));

        //Returning this allows us to make statements like "network.addconv.addmax.addfull"
        return this;
    }

    public Network addConv(int[] filterSizes, int filterCount){
        layers = ArrayUtils.appendValue(layers, new Layer_Convolutional(filterSizes, filterCount, getOutputDims()));
        return this;
    }

    //e.g. sizes {3, 3} and stride 2 applied to 5x5x5x tensor results in sizes {3, 3, 1} and strides {2, 2, 1}
    public Network addMax(int stride, int[] sizes){

        int[] strides = new int[sizes.length];
        Arrays.fill(strides, stride);

        layers = ArrayUtils.appendValue(layers, new Layer_MaxPooling( strides, sizes, getOutputDims()));
        return this;
    }

    public Network addFull(int outputLength){
        layers = ArrayUtils.appendValue(layers, new Layer_FullyConnected( outputLength, getOutputDims()));
        return this;
    }

    public Network addSoftmax(){
        layers = ArrayUtils.appendValue(layers, new Layer_SoftmaxACT(getOutputDims()));
        return this;
    }

    public Network addReLU(){
        layers = ArrayUtils.appendValue(layers, new Layer_ReluACT(getOutputDims()));
        return this;
    }

    @Override
    public Tensor forwardProp(Tensor input) {
        Tensor output = input;

        for(Layer l : layers){
            output = l.forwardProp(output);
        }

        return output;
    }

    @Override
    public Tensor backProp(Tensor outputGrad, double learningRate) {

        Tensor inputGrad = outputGrad;

        for(int i=layers.length-1; i >= 0; i--){
            inputGrad = layers[i].backProp(inputGrad, learningRate);
        }

        return inputGrad;
    }

    @Override
    public int[] getOutputDims() {
        if(layers.length == 0) return expectedInputDims;
        else return layers[layers.length - 1].getOutputDims();
    }
}
