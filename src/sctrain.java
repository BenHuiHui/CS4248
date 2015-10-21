import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.List;

/**
 * Created by huihui on 19/10/15.
 */

public class sctrain {

    //java sctrain adopt adapt adapt_adopt.train adapt_adopt.modal
    //java sctrain bought brought bought_brought.train bought_brought.modal
    //java sctrain peace piece peace_piece.train peace_piece.modal

    public static void  main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        String word1 = args[0];
        String word2 = args[1];
        String trainFilename = args[2];
        String modalFilename = args[3];

        FeatureData data = FeatureBuilder.featureDataFromFile(trainFilename, word1, word2);
        List<String> features = data.features;

        /*
        for (String f : features){
            System.out.println(f);
        }
        System.out.println(features.size());
        */

        LogisticRegression lr = new LogisticRegression(features.size());
        double[] weights = lr.train(data.instances);

        //Save features and weights to modal file
        PrintWriter writer = new PrintWriter(modalFilename, "UTF-8");

        for (String f : features) {
            writer.print(f);
            writer.print(" ");
        }

        writer.println();

        for (double w : weights){
            writer.print(w);
            writer.print(" ");
        }

        writer.close();
    }

}
