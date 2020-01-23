package uk.ac.cam.mgm52.cnn;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        //Using the MNIST dataset, which has 60,000 labelled 28x28 greyscale images. This example is just reading the first 10000.
        int[] labels = IdxReader.readLabels("resources/mnist/train-labels-idx1-ubyte", 100);
        Tensor[] labelTensors = IdxReader.labelsToTensors(labels);
        Tensor[] imageTensors = IdxReader.readGreyImages("resources/mnist/train-images-idx3-ubyte", 100);

        Network testNet = new Network(28, 28);

        testNet.addConv(new int[] {3, 3}, 8)
                .addMax(2, new int[] {2, 2})
                .addFull(10)
                .addSoftmax();

        Trainer testTrain = new Trainer(testNet, LossFunction.crossEntropy, imageTensors, labelTensors, 100,0.005);

        //Perform 12 epochs, decreasing learning rate so as to eventually focus on more subtle features in the dataset.
        for(int i = 0; i < 12; i++) {
            testTrain.epoch();
            testTrain.learningRate *= 0.82;
        }

        //Save filters as images
        ((Layer_Convolutional) testNet.layers[0]).saveFilterImages();


        //Test the network on 10,000 previously unseen images.
        int[] tenklabels = IdxReader.readLabels("resources/mnist/t10k-labels-idx1-ubyte", 10000);
        Tensor[] tenklabelTensors = IdxReader.labelsToTensors(tenklabels);
        Tensor[] tenkimageTensors = IdxReader.readGreyImages("resources/mnist/t10k-images-idx3-ubyte", 10000);

        Trainer tenktestTrain = new Trainer(testTrain.network, LossFunction.crossEntropy, tenkimageTensors, tenklabelTensors, 10000,0);

        //Learning rate is 0, so no actual training is being done here; network is testing itself.
        tenktestTrain.epoch();

    }

}