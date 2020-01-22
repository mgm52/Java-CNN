package uk.ac.cam.mgm52.cnn;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.stream.DoubleStream;

/**Stores an N-dimensional tensor. Facilitates operations including cross-correlation mapping and extracting subsets.*/
public class Tensor {
    //Length of each dimension
    public int[] dimSizes;

    //Values of all elements. Stored in Horner's scheme.
    public double[] values;

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
        this.values = new double[ArrayUtils.product(dimSizes)];
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

            String shortString = valString;
            if(shortString.length() >= 5) shortString = shortString.substring(0, Math.max(5, shortString.indexOf(".")+2));

            if(valString.contains("E")) shortString += valString.substring(valString.indexOf("E"));

            result += shortString + " ";;

            //If statement here is to prevent new lines being added to end of result
            if (i < values.length-1 && dimSizes.length > 1) {
                //Enter new line once reached end of x dimension
                if ((i + 1) % dimSizes[0] == 0) result += System.lineSeparator();
                //Enter another new line once reached end of xy plane
                if ((i + 1) % (dimSizes[0] * dimSizes[1]) == 0) result += System.lineSeparator();
            }
        }

        return result;
    }

    public String toBoolString(){
        String result = "";

        //Iterate through each value. A TensorCoordIterator is not necessary here, as we operate in indices.
        for(int i = 0; i < values.length; i++){
            //Getting next value in tensor...
            String valString = ((values[i] < 0.4) ? (values[i] < 0 ? " " : "#") : "█");

            result += valString + " ";;

            //If statement here is to prevent new lines being added to end of result
            if (i < values.length-1 && dimSizes.length > 1) {
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

    public Tensor randomsSND(){
        Tensor t = new Tensor(dimSizes);

        Random ran = new Random();

        //Set all values to random, normally distributed values with mean 0, stan dev 1
        for(int i = 0; i < t.values.length; i++){
            t.values[i] = ran.nextGaussian();
        }

        return t;
    }


    /**Get a value at some coordinates*/
    public double get(int... coords){
        int index = HornerConversion.coordsToHorner(coords, dimSizes);
        if(index >= 0 && index < values.length) return values[index];
        //Return 0 if trying to access a coordinate beyond boundaries. This is useful for padding.
        return 0;
    }

    /**Get a value at some coordinates*/
    public void set(int[] coords, double val){
        values[HornerConversion.coordsToHorner(coords, dimSizes)] = val;
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

    public boolean equals(Tensor t){
        return(Arrays.equals(t.dimSizes, dimSizes) && Arrays.equals(t.values, values));
    }

    /**Multiply each element of t1 with a corresponding element of t2, then sum these values.*/
    public double innerProduct(Tensor t){
        double result = 0;
        for(int i = 0; i < t.values.length; i++){
            result += values[i] * t.values[i];
        }
        return result;
    }

    /**Multiply each element of t1 with a corresponding element of t2.*/
    public Tensor product(Tensor t){
        return new Tensor(dimSizes, ArrayUtils.multAll(values, t.values));
    }

    public Tensor product(double val){
        return new Tensor(dimSizes, ArrayUtils.multAll(values, val));
    }

    public int maxValueIndex(){
        return ArrayUtils.findIndexOfMax(values);
    }

    public double maxValue(){
        return ArrayUtils.findIndexOfMax(values);
    }


    /**Appends one tensor to another. If base tensor rank != input rank, missing dimensions are considered to have length 1.
     * @param t input tensor to be appended.
     * @return resultant tensor
     */
    public Tensor appendTensor(Tensor t, int resultRank){
        int[] newDimSizes = Arrays.copyOf(dimSizes, resultRank);

        //Any new dimensions added need to start at 1
        for(int i = dimSizes.length; i < resultRank; i++){
            newDimSizes[i] = 1;
        }

        if(t.dimSizes.length >= resultRank){
            newDimSizes[resultRank-1] += t.dimSizes[t.dimSizes.length-1];
        }
        else{
            newDimSizes[resultRank-1] += 1;
        }

        //Append values of input tensor onto base tensor.
        double[] newValues = DoubleStream.concat(Arrays.stream(values), Arrays.stream(t.values)).toArray();

        return new Tensor(newDimSizes, newValues);
    }

    //Returns a tensor flipped diagonally.
    //Coord at dimension d -> dimsize[d] - 1 - coord[d]
    public Tensor flip(){
        Tensor flippedTensor = new Tensor(dimSizes);

        CoordUtils.CoordIterator i = new CoordUtils.CoordIterator(new int[] {0, 0, 0}, ArrayUtils.addAll(dimSizes, -1));
        while(i.hasNext()){
            int[] currentCoords = i.next();
            int[] flippedCoords = new int[currentCoords.length];
            for(int j = 0; j < currentCoords.length; j++){
                flippedCoords[j] = dimSizes[j] - 1 - currentCoords[j];
            }
            flippedTensor.set(currentCoords, get(flippedCoords));
        }
        return flippedTensor;
    }

    /**Iterates through all possible regions (of a certain size) that can be made from this tensor.
     * Can be instantiated with or without strides.*/
    class RegionsIterator implements Iterator<Tensor> {
        int[] regionSizes;
        int[] bottomCorner;
        int[] topCorner;

        CoordUtils.CoordIterInterface coordIterator;

        //Private function used by multiple similar constructors
        //Note: padding array can be any number of dimensions.
        private void setup(int[] regionSizes, int[] padding){
            int[] regionSizesCopy = regionSizes;

            //Ensure regions are the same size by appending dimensions of length 1
            while(regionSizesCopy.length < dimSizes.length) regionSizesCopy = ArrayUtils.appendValue(regionSizesCopy, 1);

            //Subtracting 1 here so that we don't have to in later iterations
            this.regionSizes = ArrayUtils.addAll(regionSizesCopy, -1);

            //Bottom corner is at origin, then displace by padding
            bottomCorner = new int[dimSizes.length];
            ArrayUtils.subtractAll(bottomCorner, padding);

            //Top corner is limited by size of region
            topCorner = ArrayUtils.subtractAll(dimSizes, regionSizesCopy);

            //But extended by padding
            topCorner = ArrayUtils.addAll(topCorner, padding);
        }

        RegionsIterator(int[] regionSizes, int[] padding){
            setup(regionSizes, padding);

            //Iterate over all coordinates that a region can be formed from
            coordIterator = new CoordUtils.CoordIterator(bottomCorner, topCorner);
        }

        //This constructor makes use of a strides array.
        RegionsIterator(int[] regionSizes, int[] padding, int[] strides){
            setup(regionSizes, padding);

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
