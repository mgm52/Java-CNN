package uk.ac.cam.mgm52.cnn;

import java.lang.reflect.Array;
import java.util.Iterator;

public class CoordUtils {

    private CoordUtils(){}


    public interface CoordIterInterface extends Iterator<int[]> {
        int[] getCurrentCoords();
        int getCurrentCount();
    }


    /**Iterates through a set of coordinates within a region.*/
    public static class CoordIterator implements CoordIterInterface {
        int[] startCoords;
        int[] finalCoords;

        //Number of coordinates that have been processed so far. Used to know when to stop.
        private int currentCount = 0;
        @Override
        public int getCurrentCount() {
            return currentCount;
        }

        private int[] currentCoords;
        @Override
        public int[] getCurrentCoords(){
            return currentCoords;
        }

        //finalIndex default at 1 because this is multiplication identity
        int finalIndex = 1;

        CoordIterator(int[] corner1, int[] corner2){
            startCoords = new int[corner1.length];
            currentCoords = new int[corner1.length];
            finalCoords = new int[corner2.length];

            //Ensure start is at minimum coords within region, end is at maximum
            for(int i = 0; i < startCoords.length; i++){
                startCoords[i] = Math.min(corner1[i], corner2[i]);
                currentCoords[i] = startCoords[i];
                finalCoords[i] = Math.max(corner1[i], corner2[i]);

                //The total number of integer coordinates within the region (not considering stride)
                finalIndex *= (finalCoords[i] - startCoords[i]) + 1;
            }
            //At each iteration, we increment the coordinate before returning it. This means that our starting position has to be 1 stride less than the first coord.
            currentCoords[0] --;
        }

        @Override
        public boolean hasNext() {
            return currentCount < finalIndex;
        }

        @Override
        public int[] next() {
            //This works like a simple counter. It increments the leftmost non-maximal number, setting any maximal numbers back to zero.
            for(int i = 0; i < currentCoords.length; i++){
                if(currentCoords[i] + 1 > finalCoords[i]){
                    currentCount += (currentCoords[i] - finalCoords[i]);
                    currentCoords[i] = startCoords[i];
                }
                else{
                    currentCoords[i] = currentCoords[i] + 1;
                    currentCount += 1;
                    //Exit the loop as soon as we've incremented a coord
                    return currentCoords;
                }
            }

            //If hasNext is working, this code should not execute.
            throw new NullPointerException("Tried to access coordinate beyond region boundary");
        }
    }




    /**Iterates through a set of coordinates within a region, with a certain stride.*/
    public static class StridingCoordIterator implements CoordIterInterface {
        int[] strides;
        int[] corner1;

        CoordIterator coordIter;

        @Override
        public int getCurrentCount() {
            return coordIter.getCurrentCount();
        }

        private int[] currentCoords;
        @Override
        public int[] getCurrentCoords(){
            return currentCoords;
        }

        StridingCoordIterator(int[] corner1, int[] corner2, int[] strides){
            this.corner1 = corner1;
            this.strides = strides;

            int[] cornerDifference = ArrayUtils.subtractAll(corner2, corner1);
            int[] adjustedDifference = ArrayUtils.divideAll(cornerDifference, strides);

            coordIter = new CoordIterator(new int[corner1.length], adjustedDifference);
        }

        @Override
        public boolean hasNext() {
            return coordIter.hasNext();
        }

        @Override
        public int[] next() {
            currentCoords = ArrayUtils.addAll(ArrayUtils.multAll(coordIter.next(), strides), corner1);
            return currentCoords;
        }
    }



}
