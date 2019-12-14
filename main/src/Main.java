public class Main {

    public static void main(String[] args) {
        Tensor testTens = new Tensor(4, 4).randoms(-1, 1);

        Tensor t = testTens.crossCorrelationMap(testTens.getRegion(new int[] {1, 0}, new int[] {3, 2}));
    }

}