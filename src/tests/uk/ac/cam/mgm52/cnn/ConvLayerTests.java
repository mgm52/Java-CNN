package uk.ac.cam.mgm52.cnn;

import org.junit.Test;
import sun.jvm.hotspot.utilities.Assert;

import java.util.Arrays;

public class ConvLayerTests {

    Tensor testTens = TestableTensors.consecutiveValues(2, 3, 4);
    Tensor smallerTestTens = TestableTensors.consecutiveValues(2, 3);

    @Test
    public void crossCorrelationMap_returnsRightValues(){
        double[] expectedMap = {2.0, 4.0, 8.0, 10.0, 14.0, 16.0, 20.0, 22.0};
        double[] ccMapResults = Layer_Convolutional.crossCorrelationMap(testTens, new Tensor(new int[] {2, 2, 1}, new double[] {0.1, 0.2, 0.3, 0.4}), new int[] {1, 2, 4}, new int[0]).values;

        Assert.that(Arrays.equals(expectedMap, ccMapResults), "crossCorrelationMap is not returning expected values.");
    }

    @Test
    public void crossCorrelationMap_returnsRightValues_withPadding(){
        double[] expectedMap = {2.0, 3.0, 4.0, 2.6, 1.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] ccMapResults = Layer_Convolutional.crossCorrelationMap(smallerTestTens, new Tensor(new int[] {2, 2}, new double[] {0.1, 0.2, 0.3, 0.4}), new int[] {3, 4}, new int[] {1, 1}).values;

        Assert.that(Arrays.equals(expectedMap, ccMapResults), "crossCorrelationMap is not returning expected values.");
    }

    @Test
    public void convLayer_backProp_doesntChangeWithZeroGradient(){
        Layer_Convolutional myConv = new Layer_Convolutional(new int[] {2, 2}, 4, new int[] {2, 3, 4});
        Tensor oldFilters = myConv.filters;

        //feed forward to set up layer for backprop
        myConv.forwardProp(testTens);

        //backprop
        Tensor zeroesGrad = new Tensor(myConv.outputDims);
        Tensor firstback = myConv.backProp(zeroesGrad, 1.0);

        Assert.that(myConv.filters.equals(oldFilters), "Backpropagating gradients=0 does not keep values the same.");
    }

    @Test
    public void convLayer_feedForward_givesExpectedOutputSize() {
        Layer_Convolutional myConv = new Layer_Convolutional(new int[] {2, 2}, 4, new int[] {2, 3, 4});
        int[] expectedOutputSizes = new int[] {1, 2, 4, 4};

        //feed forward
        Tensor forwardResult = myConv.forwardProp(testTens);

        Assert.that(Arrays.equals(forwardResult.dimSizes, expectedOutputSizes), "Feed forward on conv layer giving unexpected dim sizes");
    }
}
