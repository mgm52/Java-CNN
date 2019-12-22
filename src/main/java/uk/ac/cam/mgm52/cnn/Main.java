package uk.ac.cam.mgm52.cnn;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        maxtest();
    }

    public static void maxtest(){
        Tensor testTens = new Tensor(4, 4).randoms(-1, 1);

        int[] sizes = new int[] {2, 2};
        int[] strides = {2, 2};

        Layer_MaxPooling myMax = new Layer_MaxPooling(strides, sizes);

        System.out.println("Applying pool of size " + Arrays.toString(sizes) + " and stride " + Arrays.toString(strides) + ":");

        System.out.println("On this input:");
        System.out.println(testTens.toString());

        System.out.println("A forward pass yields this:");
        Tensor firstpass = myMax.forwardProp(testTens);
        System.out.println(firstpass.toString());

        System.out.println("Now let's imagine we got this output gradient:");
        double[] gradValues = new double[firstpass.getValuesCopy().length];
        Arrays.fill(gradValues, 1);
        Tensor testGrad = new Tensor(firstpass.getFirstDimsCopy(firstpass.rank), gradValues);
        System.out.println(testGrad.toString());

        System.out.println("We would backprop, getting these gradients:");
        Tensor firstback = myMax.backProp(testGrad, 1.0);
        System.out.println(firstback.toString());
    }

    public static void convtest(){
        Tensor testTens = new Tensor(4, 4).randoms(-1, 1);

        Layer_Convolutional myConv = new Layer_Convolutional(new int[] {3, 3}, 4);

        System.out.println("We have these filters:");
        System.out.println(myConv.filters.toString());

        System.out.println("On this input:");
        System.out.println(testTens.toString());

        System.out.println("A forward pass yields this:");
        Tensor firstpass = myConv.forwardProp(testTens);
        System.out.println(firstpass.toString());

        System.out.println("Now let's imagine we got this output gradient:");
        double[] gradValues = new double[firstpass.getValuesCopy().length];
        Arrays.fill(gradValues, 0);
        Tensor testGrad = new Tensor(firstpass.getFirstDimsCopy(firstpass.rank), gradValues);
        System.out.println(testGrad.toString());

        System.out.println("We would backprop, getting these filter gradients:");
        Tensor firstback = myConv.backProp(testGrad, 1.0);
        System.out.println(firstback.toString());

        System.out.println("And these would be our new filters:");
        System.out.println(myConv.filters.toString());
    }

}