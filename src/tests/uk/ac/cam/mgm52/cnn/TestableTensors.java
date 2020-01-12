package uk.ac.cam.mgm52.cnn;

import java.util.Arrays;
import java.util.stream.IntStream;

public class TestableTensors {

    private TestableTensors(){};

    //Populate tensor with values equal to the index of each coordinate
    public static Tensor consecutiveValues(int... dimSizes){
        int tensorSize = Arrays.stream(dimSizes).reduce(1, (i, j) -> i * j);
        double[] tensorValues = IntStream.range(0, tensorSize).mapToDouble(j -> (double) j).toArray();

        return new Tensor(dimSizes, tensorValues);
    }

}
