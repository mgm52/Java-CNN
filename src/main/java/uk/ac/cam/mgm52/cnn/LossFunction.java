package uk.ac.cam.mgm52.cnn;

/**Provides an interface to loss functions. Used in backpropagation*/
public interface LossFunction{
     Tensor calculateLoss(double[] expectedOutput, double[] actualOutput);
     Tensor calculateLossDerivative(double[] expectedOutput, double[] actualOutput);

     LossFunction crossEntropy = new LossFunction() {
        @Override
        public Tensor calculateLoss(double[] expectedOutput, double[] actualOutput) {
            double[] lossValues = new double[expectedOutput.length];

            for(int i = 0; i < lossValues.length; i++){
                if(expectedOutput[i] == 1) lossValues[i] = -1.0 * Math.log(actualOutput[i]);
                else  lossValues[i] = -1.0 * Math.log(1 - actualOutput[i]);
            }

            return new Tensor(new int[] {lossValues.length}, lossValues);
        }

        @Override
        public Tensor calculateLossDerivative(double[] expectedOutput, double[] actualOutput) {
            double[] lossDerivs = new double[expectedOutput.length];

            for(int i = 0; i < lossDerivs.length; i++){
                if(expectedOutput[i] == 1) lossDerivs[i] = -1.0 / actualOutput[i];
                else  lossDerivs[i] = 1.0 / (1 - actualOutput[i]);
            }

            return new Tensor(new int[] {lossDerivs.length}, lossDerivs);
        }
    };

}
