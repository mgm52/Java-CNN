package uk.ac.cam.mgm52.cnn;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.DoubleStream;

/**Stores an N-dimensional tensor. Facilitates operations including cross-correlation mapping and extracting subsets.*/
public class Tensor {
    //Length of each dimension
    private int[] dimSizes;

    //Values of all elements. Stored in Horner's scheme.
    private double[] values;

    //Number of dimensions.
    public final int rank;

    /**Creates new tensor with given dimensions. Default values are zero.
     * @param dimSizes the size (length) of each dimension within the tensor
     */
    public Tensor(int... dimSizes){
        //Although it would better enforce immutability, copying values would introduce unnecessary overhead, so array is assigned as a reference.
        this.dimSizes = dimSizes;
        rank = dimSizes.length;

        //The length of our 1-dimensional values array needs to be equivalent to the product of all dimensions
        this.values = new double[Arrays.stream(dimSizes).reduce(1, (i, j) -> i * j)];
    }

    /**Creates new tensor with given dimensions, values.
     * @param dimSizes the size (length) of each diminsion within the tensor
     * @param values all values to be used by the tensor, given using horner's scheme
     */
    public Tensor(int[] dimSizes, double[] values){
        //Although it would better enforce immutability, copying values would introduce unnecessary overhead, so arrays are assigned as references.
        this.dimSizes = dimSizes;
        rank = dimSizes.length;
        this.values = values;
    }

    /**Return copy of Tensor values array. This should not be used at all frequently: copying all values is costly.*/
    public double[] getValuesCopy(){
        return Arrays.copyOf(values, values.length);
    }

    /**Returns a subset of the dimensions of the tensor, starting from dim 0
     * @param length The number of dimensions to return
     * @return
     */
    public int[] getFirstDimsCopy(int length){
        return Arrays.copyOf(dimSizes, length);
    }

    /**
     * Return string displaying all tensor values (as a sequence of matrices).
     * Each matrix represents an XY plane within tensor. Starts with lowest z, w, etc values.
     */
    public String toString(){
        String result = "";

        //Iterate through each value. A TensorCoordIterator is not necessary here, as we operate in indices.
        for(int i = 0; i < values.length; i++){
            //Getting next value in tensor...
            String valString = values[i]+"";

            //Format number to reduce length
            //(Note: += here performs poorly as it creates a new object each time it is called)
            //(Could remedy this with StringBuilder, but performance < readability in this function)
            if(valString.length() >= 5) valString = valString.substring(0, Math.max(5, valString.indexOf(".")+2));
            result += valString + " ";;

            //If statement here is to prevent new lines being added to end of result
            if (i < values.length-1) {
                //Enter new line once reached end of x dimension
                if ((i + 1) % dimSizes[0] == 0) result += System.lineSeparator();
                //Enter another new line once reached end of xy plane
                if ((i + 1) % (dimSizes[0] * dimSizes[1]) == 0) result += System.lineSeparator();
            }
        }

        return result;
    }


    /**Return new tensor with same dimensions, but values at zero.*/
    public Tensor zeroes(){
        return new Tensor(dimSizes);
    }

    /**Return new tensor with same dimensions, but values randomized within range.
     * @param min min value of each item
     * @param max max value of each item
     */
    public Tensor randoms(double min, double max){
        Tensor t = new Tensor(dimSizes);

        //Set all values to random within range [min, max)
        for(int i = 0; i < t.values.length; i++){
            t.values[i] = Math.random() * (max-min) + min;
        }

        return t;
    }


    /**Get a value at a coordinate*/
    public double get(int... coords){
        return values[HornerConversion.coordsToHorner(coords, dimSizes)];
    }

    /**Returns tensor with opposite corners at corner1 and corner2
     * @param corner1 First corner of region. A list of integer coordinates.
     * @param corner2 Second corner of region. A list of integer coordinates.
     */
    public Tensor getRegion(int[] corner1, int[] corner2){
        //Lengths of new region dimensions = difference in lengths between the two corners plus one.
        int[] newDimSizes = new int[corner1.length];
        for(int i = 0; i < newDimSizes.length; i++){
            newDimSizes[i] = Math.abs(corner2[i] - corner1[i]) + 1;
        }
        //Create new tensor representing the region...
        Tensor region = new Tensor(newDimSizes);

        //Use iterator to assign each value
        for (CoordUtils.CoordIterator i = new CoordUtils.CoordIterator(corner1, corner2); i.hasNext(); ) {
            int[] coords = i.next();
            //Iterator's currentIndex value corresponds to index within region
            region.values[i.getCurrentCount()-1] = get(coords);
        }

        return region;
    }

    /**Elementwise addition between two tensors. Each element of t is multiplied by "factor" first.*/
    public Tensor add(Tensor t, double factor){
        double[] newVals = new double[values.length];
        for(int i = 0; i < values.length; i ++){
            newVals[i] = values[i] + t.values[i] * factor;
        }
        return new Tensor(dimSizes, newVals);
    }

    /**Multiply each element of t1 with a corresponding element of t2, then sum these values.*/
    public double innerProduct(Tensor t){
        double result = 0;
        for(int i = 0; i < t.values.length; i++){
            result += values[i] * t.values[i];
        }
        return result;
    }

    /**Generate cross-correlation map of a filter applied to this tensor.*/
    public Tensor crossCorrelationMap(Tensor filter){
        //The size of the resulting map = base size - (filter size - 1) in each dimension
        Tensor ccMap = new Tensor(calculateMapSize(filter.dimSizes));

        for (TensorRegionsIterator i = new TensorRegionsIterator(filter.dimSizes); i.hasNext(); ) {
            Tensor region = i.next();
            ccMap.values[i.coordIterator.getCurrentCount()-1] = region.innerProduct(filter);
        }

        return ccMap;
    }

    /**Calculate the size of the cross correlation map resultant from applying a given filter*/
    public int[] calculateMapSize(int[] filterDimSizes){
        return ArrayUtils.subtractAll(dimSizes, ArrayUtils.addAll(filterDimSizes, -1));
    }

    public double maxValue(){
        double max = values[0];
        for(int i = 1; i < values.length; i++){
            if (values[i] > max) max = values[i];
        }

        return max;
    }


    /**Appends one tensor to another. Base tensor must be of equal or one-higher rank to input tensor.
     * e.g. (3, 3) tensor append (3, 3) tensor = (3, 3, 2) tensor
     *      (4, 4, 3) tensor append (4, 4) tensor = (4, 4, 4) tensor
     * @param t input tensor to be appended. of equal or one-less rank to base tensor.
     * @return resultant tensor
     */
    public Tensor appendTensor(Tensor t){

        int[] newDimSizes;
        if(t.dimSizes.length < dimSizes.length){
            //e.g. (4, 4, 3) append with (4, 4) tensor. Should result in (4, 4, 4).
            newDimSizes = Arrays.copyOf(dimSizes, dimSizes.length);
            newDimSizes[newDimSizes.length - 1]++;
        }
        else{
            //e.g. (3, 3) append with (3, 3) tensor. Should result in (3, 3, 2).
            newDimSizes = ArrayUtils.appendValue(dimSizes, 2);
        }

        //Add values of input tensor onto base tensor.
        double[] newValues = DoubleStream.concat(Arrays.stream(values), Arrays.stream(t.values)).toArray();

        return new Tensor(newDimSizes, newValues);
    }



    /**Iterates through all possible regions (of a certain size) that can be made from this tensor.
     * Can be instantiated with or without strides.*/
    class TensorRegionsIterator implements Iterator<Tensor> {
        int[] regionSizes;
        int[] bottomCorner;
        int[] topCorner;

        CoordUtils.CoordIterInterface coordIterator;

        //Private function used by multiple similar constructors
        private void setup(int[] regionSizes){
            //Ensure regions are the same size by appending dimensions of length 1
            while(regionSizes.length < dimSizes.length) regionSizes = ArrayUtils.appendValue(regionSizes, 1);

            //Subtracting 1 here so that we don't have to in later iterations
            this.regionSizes = ArrayUtils.addAll(regionSizes, -1);

            //Bottom corner is at origin
            bottomCorner = new int[dimSizes.length];

            //Top corner is limited by size of region
            topCorner = ArrayUtils.subtractAll(dimSizes, regionSizes);
        }

        TensorRegionsIterator(int[] regionSizes){
            setup(regionSizes);

            //Iterate over all coordinates that a region can be formed from
            coordIterator = new CoordUtils.CoordIterator(bottomCorner, topCorner);
        }

        //This constructor makes use of a strides array.
        TensorRegionsIterator(int[] regionSizes, int[] strides){
            setup(regionSizes);

            //Iterate over all coordinates that a region can be formed from
            coordIterator = new CoordUtils.StridingCoordIterator(bottomCorner, topCorner, strides);
        }

        @Override
        public boolean hasNext() {
            return coordIterator.hasNext();
        }

        @Override
        public Tensor next() {
            int[] regionBottomCorner = coordIterator.next();
            int[] regionTopCorner = ArrayUtils.addAll(regionBottomCorner, regionSizes);
            return getRegion(regionBottomCorner, regionTopCorner);
        }
    }



}
