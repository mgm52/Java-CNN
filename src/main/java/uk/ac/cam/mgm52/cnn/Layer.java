package uk.ac.cam.mgm52.cnn;

public interface Layer {

    public Tensor forwardProp(Tensor input);
    public Tensor backProp(Tensor outputGrad, double learningRate);
    public int[] getOutputDims();

}