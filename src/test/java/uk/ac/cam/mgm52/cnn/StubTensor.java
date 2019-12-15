package uk.ac.cam.mgm52.cnn;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class StubTensor extends Tensor {

    StubTensor(int[] dimSizes){
        //super(new int[3], IntStream.range(0, Arrays.stream(dimSizes).reduce(1, (i, j) -> i * j)).mapToDouble(j -> (double) j).toArray());
    }

}
