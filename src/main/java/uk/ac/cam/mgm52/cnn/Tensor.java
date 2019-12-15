package uk.ac.cam.mgm52.cnn;

import java.util.Arrays;
import java.util.Iterator;

public class Tensor {
    //Length of each dimension
    private int[] dimSizes;

    //Values of all elements. Stored in Horner's scheme.
    private double[] values;

    /**Creates new tensor with given dimensions. Default values are zero.
     * @param dimSizes the size (length) of each diminsion within the tensor
     */
    public Tensor(int... dimSizes){
        //Copying values would introduce unnecessary overhead, so array is assigned as a reference
        this.dimSizes = dimSizes;

        //The length of our 1-dimensional values array needs to be equivalent to the product of all dimensions
        this.values = new double[Arrays.stream(dimSizes).reduce(1, (i, j) -> i * j)];
    }

    /**Creates new tensor with given dimensions, values.
     * @param dimSizes the size (length) of each diminsion within the tensor
     * @param values all values to be used by the tensor, given using horner's scheme
     */
    public Tensor(int[] dimSizes, double[] values){
        //Copying values would introduce unnecessary overhead, so arrays are assigned as references
        this.dimSizes = dimSizes;
        this.values = values;
    }

    /**Return copy of Tensor values array. This should not be used at all frequently: copying all values is costly.*/
    public double[] getValuesCopy(){
        return Arrays.copyOf(values, values.length);
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
            result += valString.substring(0, Math.max(5, valString.indexOf(".")+2)) + " ";

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
        Tensor t = new Tensor(dimSizes);
        return t;
    }

    /**Return new tensor with same dimensions, but values randomized within range.
     * @param min min value of each item
     * @param max max value of each item
     */
    public Tensor randoms(float min, float max){
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
        for (TensorCoordIterator i = new TensorCoordIterator(corner1, corner2); i.hasNext(); ) {
            int[] coords = i.next();
            //Iterator's currentIndex value corresponds to index within region
            region.values[i.currentIndex] = get(coords);
        }

        return region;
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
        Tensor ccMap = new Tensor(ArrayUtils.subtractAll(dimSizes, ArrayUtils.addAll(filter.dimSizes, -1)));

        for (TensorRegionsIterator i = new TensorRegionsIterator(filter.dimSizes); i.hasNext(); ) {
            Tensor region = i.next();
            ccMap.values[i.coordIterator.currentIndex] = region.innerProduct(filter);
        }

        return ccMap;
    }


    /**Iterates through all possible regions (of a certain size) that can be made from this tensor.*/
    class TensorRegionsIterator implements Iterator<Tensor>{
        int[] regionSizes;

        int[] bottomCorner;
        int[] topCorner;
        TensorCoordIterator coordIterator;

        TensorRegionsIterator(int[] regionSizes){
            //Subtracting 1 here so that we don't have to in later iterations
            this.regionSizes = ArrayUtils.addAll(regionSizes, -1);

            //Bottom corner is at origin
            bottomCorner = new int[dimSizes.length];

            //Top corner is limited by size of region
            topCorner = ArrayUtils.subtractAll(dimSizes, regionSizes);

            //Iterate over all coordinates that a region can be formed from
            coordIterator = new TensorCoordIterator(bottomCorner, topCorner);

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

    /**Iterates through a set of coordinates within a region.*/
    class TensorCoordIterator implements Iterator<int[]> {
        int[] startCoords;
        int[] finalCoords;

        int[] currentCoords;

        //Number of coordinates that have been processed so far. Used to know when to stop. Starts at -1 so that it corresponds to local index in horner scheme.
        int currentIndex = -1;
        //finalIndex default at 1 because this is multiplication identity
        int finalIndex = 1;

        TensorCoordIterator(int[] corner1, int[] corner2){
            startCoords = new int[corner1.length];
            currentCoords = new int[corner1.length];
            finalCoords = new int[corner2.length];

            //Ensure start is at minimum coords within region, end is at maximum
            for(int i = 0; i < startCoords.length; i++){
                startCoords[i] = Math.min(corner1[i], corner2[i]);
                currentCoords[i] = startCoords[i];
                finalCoords[i] = Math.max(corner1[i], corner2[i]);

                //The total number of integer coordinates within the region
                finalIndex *= (finalCoords[i] - startCoords[i]) + 1;
            }
            //At each iteration, we increment the coordinate before returning it. This means that our starting position has to be 1 less than the first coord.
            currentCoords[0] --;
        }

        @Override
        public boolean hasNext() {
            return currentIndex+1 < finalIndex;
        }

        @Override
        public int[] next() {
            //This works like a simple counter. It increments the leftmost non-maximal number, setting any maximal numbers back to zero.
            //Note: using this instead of hornerToCoords for the sake of efficiency.
            for(int i = 0; i < currentCoords.length; i++){
                currentCoords[i] = currentCoords[i];

                if(currentCoords[i] == finalCoords[i]){
                    currentCoords[i] = startCoords[i];
                }
                else{
                    currentCoords[i] = currentCoords[i] + 1;
                    currentIndex++;
                    //Exit the loop as soon as we've incremented a coord
                    return currentCoords;
                }
            }

            //If hasNext is working, this code should not execute.
            throw new NullPointerException("Tried to access coordinate beyond region boundary");
        }
    }

}
