import java.util.List;

public class Instance {

    public List<Integer> features;
    public int label;

    public Instance(List<Integer>features, int label){
        this.features = features;
        this.label = label;
    }
}
