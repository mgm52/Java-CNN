package uk.ac.cam.mgm52.cnn;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

//Given a set of layers, a set of input data, and a set of expected output data, it
public class Trainer {

    Network network;

    Random rand = new Random();

    LossFunction lossFun;

    Tensor[] inputs;
    Tensor[] labels;

    double learningRate;

    int talkInterval = 500;

    int epochCount = 0;

    public Trainer(Network network, LossFunction lossFun, Tensor[] inputs, Tensor[] labels, int talkInterval, double learningRate){
        this.network = network;
        this.lossFun = lossFun;
        this.inputs = inputs;
        this.labels = labels;
        this.talkInterval = talkInterval;
        this.learningRate = learningRate;
    }


    public void epoch(){
        epochCount++;
        double averageLoss = 0;
        double accuracy = 0;

        say("BEGINNING EPOCH " + epochCount);

        //The order in which the trainer reads inputs/labels in this epoch. Randomized.
        int[] trainOrder = ArrayUtils.randomOrderInts(0, inputs.length - 1);

        for(int i = 0; i < trainOrder.length; i++){
            int j = trainOrder[i];
            double[] recentOutput = train(inputs[j], labels[j]);

            if(talkInterval>0) {
                averageLoss += ArrayUtils.sum(lossFun.calculateLoss(labels[j].values, recentOutput).values);

                if(ArrayUtils.findIndexOfMax(labels[j].values) == ArrayUtils.findIndexOfMax(recentOutput)){accuracy++;}

                if((i+1) % talkInterval == 0){
                    say("At input " + (i+1) + " / " + trainOrder.length + ", average loss for last " + talkInterval + " iterations is " + averageLoss/talkInterval);
                    say("and accuracy is " + 100 * accuracy/talkInterval + "%");
                    say("(learning rate " + learningRate  + ")");

                    averageLoss = 0;
                    accuracy = 0;
                }
            }
        }

        say("Completed epoch");
        say("");
    }

    public void say(String message){
        if(talkInterval>0) System.out.println(message);
    }

    public void printExamples(int n){
        for(int i = 0; i < n; i++){
            int r = rand.nextInt(inputs.length);
            Tensor output = network.forwardProp(inputs[r]);

            System.out.println("Given input:");
            System.out.println(inputs[r].toBoolString());

            System.out.println("The network predicted " + ArrayUtils.findIndexOfMax(output.values));
            System.out.println("The actual value was " + ArrayUtils.findIndexOfMax(labels[r].values));

        }
    }

    public double[] train(Tensor input, Tensor label){
        Tensor output = network.forwardProp(input);
        network.backProp(lossFun.calculateLossDerivative(label.values, output.values), learningRate);

        //System.out.println("Label: " + label.toString());
        //System.out.println(input.toBoolString());

        return output.values;
    }



}
