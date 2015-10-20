import java.util.List;

public class Instance {

    public int[] features;
    public int label;

    public Instance(){
        //Do nothing
    }

    public Instance(int[]features, int label){
        this.features = features;
        this.label = label;
    }
}
