package uk.ac.cam.mgm52.cnn;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sun.jvm.hotspot.utilities.Assert;
import uk.ac.cam.mgm52.cnn.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;

@RunWith(JUnit4.class)
public class TensorTests {

    Tensor testTensor = TestableTensors.consecutiveValues(2, 3, 4);

    @Test
    public void canGetCoord_Zero(){
        Assert.that(testTensor.get(0, 0, 0) == 0, "Could not get correct value at coord (0, 0, 0)");
    }

    @Test
    public void canGetCoord_InThirdDimension(){
        Assert.that(testTensor.get(0, 0, 2) == 12, "Could not get correct value at coord (0, 0, 2)");
    }

    @Test
    public void canGetCoord_Final(){
        Assert.that(testTensor.get(1, 2, 3) == 23, "Could not get correct value at coord (1, 2, 3) (last coord)");
    }

    @Test
    public void canGetCoord_Large(){
        Tensor largeTestTensor = TestableTensors.consecutiveValues(3, 5, 2, 3, 6 ,8, 2, 3);

        Assert.that(largeTestTensor.get(0, 4, 1, 2, 5, 7, 1, 2) == 8640 * 3 - 3, "Could not get correct value at large coord (0, 4, 1, 2, 5, 7, 1, 2)");
    }

    @Test
    public void randoms_areDifferent(){
        Tensor randTestTensor = testTensor.randoms(0, 1);
        Tensor randTestTensor2 = testTensor.randoms(0, 1);

        Assert.that(!(Arrays.equals(randTestTensor.values, randTestTensor2.values)), "Randomly initialised tensors have identical values.");
    }

    @Test
    public void coordsIterator_returnsRightvalues(){
        //Expected values
        int[][] allCoords = {{0, 0, 0}, {1, 0, 0},
                             {0, 1, 0}, {1, 1, 0},
                             {0, 2, 0}, {1, 2, 0},

                             {0, 0, 1}, {1, 0, 1},
                             {0, 1, 1}, {1, 1, 1},
                             {0, 2, 1}, {1, 2, 1},

                             {0, 0, 2}, {1, 0, 2},
                             {0, 1, 2}, {1, 1, 2},
                             {0, 2, 2}, {1, 2, 2},

                             {0, 0, 3}, {1, 0, 3},
                             {0, 1, 3}, {1, 1, 3},
                             {0, 2, 3}, {1, 2, 3}};

        //Populate list with iterator values
        CoordUtils.CoordIterator ti = new CoordUtils.CoordIterator(new int[] {0, 0, 0}, new int[] {1, 2, 3});
        ArrayList<int[]> allIteratorCoords = new ArrayList<>();
        ti.forEachRemaining(c -> allIteratorCoords.add(Arrays.copyOf(c, c.length)));

        Assert.that(Arrays.deepEquals(allCoords, allIteratorCoords.toArray()), "Tensor coord iterator is not returning the correct values.");
    }


    @Test
    public void regionsIterator_returnsRightvalues(){
        //Expected values
        double[][] allRegionValues = {{0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0},
                                {2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 11.0},
                                {6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 15.0},
                                {8.0, 9.0, 10.0, 11.0, 14.0, 15.0, 16.0, 17.0},
                                {12.0, 13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 21.0},
                                {14.0, 15.0, 16.0, 17.0, 20.0, 21.0, 22.0, 23.0}};

        //Populate list with iterator values
        Tensor.RegionsIterator ti = testTensor.new RegionsIterator(new int[] {2, 2, 2}, new int[0]);
        ArrayList<double[]> allIteratorRegionValues = new ArrayList<>();
        ti.forEachRemaining(v -> allIteratorRegionValues.add(Arrays.copyOf(v.values, v.values.length)));

        Assert.that(Arrays.deepEquals(allRegionValues, allIteratorRegionValues.toArray()), "Tensor region iterator is not returning the correct values.");
    }

    @Test
    public void regionsIterator_returnsSameTensorForFullSize(){
        //Populate list with iterator values
        Tensor.RegionsIterator ti = testTensor.new RegionsIterator(new int[] {2, 3, 4}, new int[0]);
        ArrayList<double[]> allIteratorRegionValues = new ArrayList<>();
        ti.forEachRemaining(v -> allIteratorRegionValues.add(Arrays.copyOf(v.values, v.values.length)));

        Assert.that(Arrays.equals(testTensor.values, allIteratorRegionValues.get(0)), "Tensor region iterator is not returning a copy of the full tensor when given a full sized region.");
    }

    @Test
    public void getRegion_returnsSameTensorForFullSize(){
        double[] foundRegion = testTensor.getRegion(new int[] {0, 0, 0}, new int[] {1, 2, 3}).values;

        Assert.that(Arrays.equals(testTensor.values, foundRegion), "Tensor getRegion is not returning a copy of the full tensor when given a full sized region.");
    }

    @Test
    public void getRegion_returnsRightValues(){
        double[] expectedRegion = new double[] {9.0, 11.0, 15.0, 17.0};

        double[] foundRegion = testTensor.getRegion(new int[] {1, 1, 1}, new int[] {1, 2, 2}).values;

        Assert.that(Arrays.equals(expectedRegion, foundRegion), "Tensor getRegion is not returning expected values.");
    }

    @Test
    public void innerProduct_returnsRightValues(){
        double foundProd = testTensor.innerProduct(testTensor);

        Assert.that(foundProd == 4324.0, "Tensor innerProduct is not returning expected values.");
    }


}
