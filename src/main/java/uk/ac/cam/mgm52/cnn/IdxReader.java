package uk.ac.cam.mgm52.cnn;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;

/**Reads IDX files*/
public class IdxReader {

    private IdxReader(){}

    //Adapted from Radek Mackowiak's method https://stackoverflow.com/a/20383900
    public static Tensor[] readGreyImages(String path, int limit) throws IOException {
        FileInputStream inImage;

        inImage = new FileInputStream(path);

        int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
        int numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
        int numberOfRows  = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
        int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

        int numberOfPixels = numberOfRows * numberOfColumns;

        Tensor[] images = new Tensor[limit];

        for(int i = 0; i < limit; i++) {

            double[] imgPixels = new double[numberOfPixels];

            if(i % 100 == 0) {System.out.println("Number of images extracted: " + i);}

            for(int p = 0; p < numberOfPixels; p++) {
                double grey = (128 - inImage.read())/255.0;
                imgPixels[p] = grey;
            }

            images[i] = new Tensor(new int[] {numberOfColumns, numberOfRows}, imgPixels);
        }

        return images;
    }

    public static int[] readLabels(String path, int limit) throws IOException {
        FileInputStream inLabel = new FileInputStream(path);

        int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
        int numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

        int[] labels = new int[limit];

        for(int i = 0; i < limit; i++) {
           labels[i] = inLabel.read();
       }

        return labels;
    }

    public static Tensor[] labelsToTensors(int[] labels){
        //Determine unique labels
        ArrayList<Integer> uniqueLabels = new ArrayList<>();
        Tensor[] labelsTensors = new Tensor[labels.length];

        for(int l : labels){
            if(!uniqueLabels.contains(l)) uniqueLabels.add(l);
        }

        uniqueLabels.sort(Integer::compareTo);

        for(int i = 0; i < labelsTensors.length; i++){
            double[] vals = new double[uniqueLabels.size()];

            int labelID = uniqueLabels.indexOf(labels[i]);
            vals[labelID] = 1.0;

            labelsTensors[i] = new Tensor(new int[] {uniqueLabels.size()}, vals);
        }

        return labelsTensors;
    }
}