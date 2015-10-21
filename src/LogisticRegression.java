//import java.util.ArrayList;
import java.util.List;

public class LogisticRegression {

    private double[] weights;
    private double rate = 0.0001;
    //private double threshold = 0.01;
    private int numOfIterations = 5000;

    public LogisticRegression(int n){
        this.weights = new double[n];

        //Initialize
        for (int i = 0; i < n; i++) {
            this.weights[i] = 0.0;
        }
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

        System.out.println("Training");

        for (int i = 0; i < numOfIterations; i++) {

            //double error = 0.0;

            //Batch gradient descent
            /*
            for (int j = 0; j < weights.length; j++) {

                double dir = 0.0;

                for (int k = 0; k < instances.size(); k++) {

                    Instance instance = instances.get(k);
                    double predicted = classify(instance);
                    int label = instance.label;
                    dir += (label - predicted) * instance.features[k];
                }

                weights[j] = weights[j] + rate / instances.size() * dir;
                error += dir * dir;
            }
            */


            //Stochastic gradient descent

            double[] error = new double[weights.length];

            for (int j = 0; j < instances.size(); j++) {

                Instance instance = instances.get(j);
                double predicted = classify(instance);
                int label = instance.label;
                for (int k=0; k<weights.length; k++) {
                    double dir = (label - predicted) * instance.features[k];
                    weights[k] = weights[k] + rate * dir;
                    //error += dir * dir;
                    error[k] += dir;
                }
            }


            //Stop when error is smaller than the predefined error
            double err = 0;
            for (int j = 0; j < error.length; j++) {
                err += error[j] * error[j];
            }
            err = Math.sqrt(err);

            if (err < 1) break;
            //System.out.println(err);
        }

        return weights;
    }

    /*
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
    */

    public int classifySingle(Instance instance){
        //System.out.println(classify(instance));
        return classify(instance) > 0.5 ? 1 : 0;
    }

    private double classify (Instance instance){

        double logit = .0;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * instance.features[i];
        }
        return sigmoid(logit);
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }
}
