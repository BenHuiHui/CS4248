import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by huihui on 19/10/15.
 */
public class sctest {

    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        //java sctest adopt adapt adapt_adopt.test adapt_adopt.modal adapt_adopt.answer.test
        //java sctest bought brought bought_brought.test bought_brought.modal bought_brought.answer.test

        String word1 = args[0];
        String word2 = args[1];
        String testFilename = args[2];
        String modalFilename = args[3];
        String answerFilename = args[4];

        List<String> features = new ArrayList<String>();
        Map<String, Integer> indexes = new HashMap<String, Integer>();
        double[] weights = new double[0];

        //Read in the modal file
        try {

            FileReader fileReader = new FileReader(modalFilename);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;
            int i = 0;

            while((line = bufferedReader.readLine()) != null) {
                line = line.replace("\n", "").replace("\r", "");
                //System.out.println(line);

                //features
                if (i==0){
                    String[] feats = line.split(" ");

                    weights = new double[feats.length];

                    for (int j = 0; j < feats.length; j++) {
                        features.add(feats[j]);
                        indexes.put(feats[j], j);
                    }
                }

                //weights
                else{

                    String[] ws = line.split(" ");

                    for (int j = 0; j < ws.length; j++) {
                        weights[j] = Double.parseDouble(ws[j]);
                    }

                }

                i++;
            }

        }
        catch(FileNotFoundException ex) {
            System.out.println("Unable to open file for stopwords");
        }
        catch(IOException ex) {
            System.out.println("Error reading file for stopwords");
        }

        //printDouble(weights);
        //System.out.println();
        //printString(features);

        LogisticRegression lr = new LogisticRegression(weights);

        PrintWriter writer = new PrintWriter(answerFilename, "UTF-8");

        //Generate feature vectors
        try {

            FileReader fileReader = new FileReader(testFilename);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;
            while((line = bufferedReader.readLine()) != null) {

                line = line.replace("\n", "").replace("\r", "");
                String[] sent = line.split("\t");
                String id = sent[0];
                String[] words = sent[1].split(" ");

                int[] values = new int[weights.length];

                for (String word : words){
                    String f = word.toLowerCase();

                    if (indexes.containsKey(f)){
                        values[indexes.get(f)] ++;
                    }
                }

                Instance i = new Instance();
                i.features = values;
                int label = lr.classifySingle(i);

                writer.print(id);
                writer.print("\t");
                if (label == 0)
                    writer.print(word1);
                else
                    writer.print(word2);

                writer.println();
            }

        }
        catch(FileNotFoundException ex) {
            System.out.println("Unable to open file for stopwords");
        }
        catch(IOException ex) {
            System.out.println("Error reading file for stopwords");
        }

        writer.close();
    }

    private static void printList(List<Object> lists){
        for(Object o : lists){
            System.out.print(o);
            System.out.print(" ");
        }
        System.out.println();
    }

    private static void printDouble(double[] lists){
        for(double o : lists){
            System.out.print(o);
            System.out.print(" ");
        }
        System.out.println();
    }

    private static void printString(List<String> lists){
        for(String o : lists){
            System.out.print(o);
            System.out.print(" ");
        }
        System.out.println();
    }
}
