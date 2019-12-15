package uk.ac.cam.mgm52.cnn;

public class Main {

    public static void main(String[] args) {
        Tensor testTens = new Tensor(4, 4).randoms(-1, 1);

        System.out.println(testTens.toString());
    }

}