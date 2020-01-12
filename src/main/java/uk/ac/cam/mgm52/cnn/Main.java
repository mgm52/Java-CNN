package uk.ac.cam.mgm52.cnn;

import java.awt.image.ConvolveOp;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws IOException {
        int[] labels = IdxReader.readLabels("resources/kuzushiji/train-labels-idx1-ubyte", 6000);
        Tensor[] labelTensors = IdxReader.labelsToTensors(labels);
        Tensor[] imageTensors = IdxReader.readGreyImages("resources/kuzushiji/train-images-idx3-ubyte", 6000);

        Network testNet = new Network(28, 28);
        //testNet.addConv(3, 8).addMax(2, new int[] {2, 2}).addFull(10).addSoftmax();
        testNet.addConv(new int[] {3, 3}, 8)
                .addMax(2, new int[] {2, 2})
                .addFull(10).addSoftmax();


        Trainer testTrain = new Trainer(testNet, LossFunction.crossEntropy, imageTensors, labelTensors, 100,0.005);

        for(int i = 0; i < 12; i++) {

            testTrain.epoch();
            testTrain.learningRate *= 0.82;

        }

        int[] tenklabels = IdxReader.readLabels("resources/kuzushiji/t10k-labels-idx1-ubyte", 10000);
        Tensor[] tenklabelTensors = IdxReader.labelsToTensors(tenklabels);
        Tensor[] tenkimageTensors = IdxReader.readGreyImages("resources/kuzushiji/t10k-images-idx3-ubyte", 10000);

        Trainer tenktestTrain = new Trainer(testTrain.network, LossFunction.crossEntropy, tenkimageTensors, tenklabelTensors, 10000,0.005);

        tenktestTrain.epoch();
    }

    public static void fullyconnectedtest(){

        int outputlength = 5;
        int[] inputdimensions = new int[] {3, 3};
        Tensor testTens = new Tensor(inputdimensions).randoms(-1, 1);

        Layer_FullyConnected myFC = new Layer_FullyConnected(outputlength, inputdimensions);

        System.out.println("Applying these weights:");
        System.out.println(myFC.weights.toString());

        System.out.println("On this input:");
        System.out.println(testTens.toString());

        System.out.println("A forward pass yields this:");
        Tensor firstpass = myFC.forwardProp(testTens);
        System.out.println(firstpass.toString());

        System.out.println("Now let's imagine we got this output gradient:");
        double[] gradValues = new double[firstpass.values.length];
        Arrays.fill(gradValues, 0);
        Tensor testGrad = new Tensor(firstpass.getFirstDimsCopy(firstpass.rank), gradValues);
        System.out.println(testGrad.toString());

        System.out.println("We would backprop, getting these gradients:");
        Tensor firstback = myFC.backProp(testGrad, 1.0);
        System.out.println(firstback.toString());

        System.out.println("And these would be our new weights:");
        System.out.println(myFC.weights.toString());
    }

    public static void maxtest(){
        Tensor testTens = new Tensor(4, 4).randoms(-1, 1);

        int[] sizes = new int[] {2, 2};
        int[] strides = {2, 2};

        Layer_MaxPooling myMax = new Layer_MaxPooling(strides, sizes, new int[] {4, 4});

        System.out.println("Applying pool of size " + Arrays.toString(sizes) + " and stride " + Arrays.toString(strides) + ":");

        System.out.println("On this input:");
        System.out.println(testTens.toString());

        System.out.println("A forward pass yields this:");
        Tensor firstpass = myMax.forwardProp(testTens);
        System.out.println(firstpass.toString());

        System.out.println("Now let's imagine we got this output gradient:");
        double[] gradValues = new double[firstpass.values.length];
        Arrays.fill(gradValues, 1);
        Tensor testGrad = new Tensor(firstpass.getFirstDimsCopy(firstpass.rank), gradValues);
        System.out.println(testGrad.toString());

        System.out.println("We would backprop, getting these gradients:");
        Tensor firstback = myMax.backProp(testGrad, 1.0);
        System.out.println(firstback.toString());
    }

    public static void convtest(){

    }

}