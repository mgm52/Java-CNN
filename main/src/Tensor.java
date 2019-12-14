import java.util.Arrays;

public class Tensor {
    //Length of each dimension
    private int[] dimSizes;

    //Values of all elements. Stored in Horner's scheme format.
    private double[] values;

    //Creates new tensor with given dimensions. Default values are zero.
    public Tensor(int... dimSizes){
        //Copying values would introduce unnecessary overhead, so array is assigned as a reference
        this.dimSizes = dimSizes;

        //The length of our 1-dimensional values array needs to be equivalent to the product of all dimensions
        this.values = new double[Arrays.stream(dimSizes).reduce(1, (i, j) -> i * j)];
    }


    //Return new tensor with same dimensions, but values at zero
    public Tensor zeroes(){
        Tensor t = new Tensor(dimSizes);
        return t;
    }

    //Return new tensor with same dimensions, but values randomized within range
    public Tensor randoms(float min, float max){
        Tensor t = new Tensor(dimSizes);

        //Set all values to random within range [min, max)
        for(int i = 0; i < t.values.length; i++){
            t.values[i] = Math.random() * (max-min) + min;
        }

        return t;
    }

}
