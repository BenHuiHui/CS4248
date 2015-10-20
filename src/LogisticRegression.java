import java.util.ArrayList;
import java.util.List;

public class LogisticRegression {

    private double[] weights;
    private double rate = 0.0001;
    //private double threshold = 0.01;
    private int numOfIterations = 30000;

    public LogisticRegression(int n){
        this.weights = new double[n];
    }

    public LogisticRegression(double[] weights){
        this.weights = weights;
    }

    /**
     * Train logistic regression
     * @param instances training data
     * @return trained weights
     */
    public double[] train (List<Instance>instances){

        for (int i = 0; i < numOfIterations; i++) {

            //Stochastic gradient descent
            for (int j = 0; j < instances.size(); j++) {

                Instance instance = instances.get(j);
                double predicted = classify(instance);
                int label = instance.label;
                for (int k=0; k<weights.length; k++) {
                    weights[k] = weights[k] + rate * (label - predicted) * instance.features.get(k);
                }
            }
        }

        return weights;
    }

    public List<Integer> classify (List<Instance> instances){
        List<Integer> results = new ArrayList<Integer>();

        for (Instance instance: instances){
            if (classify(instance) >= 0.5){
                results.add(1);
            }
            else{
                results.add(0);
            }
        }

        return results;
    }

    private double classify (Instance instance){

        double logit = .0;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * instance.features.get(i);
        }
        return sigmoid(logit);
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }
}
