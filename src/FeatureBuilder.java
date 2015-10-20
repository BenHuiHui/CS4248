import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;


public class FeatureBuilder {

    private static final int left_win_size = 0;
    private static final int right_win_size = 0;

    static FeatureData featureDataFromFile(String fileName, String word1, String word2){

        FeatureData data = new FeatureData();

        List<String> stopwords = stopwords();
        List<String> sentences = new ArrayList<String>();
        List<String> features = new ArrayList<String>();
        Map<String, Integer> indexes = new HashMap<String, Integer>();

        //Read file
        try {

            FileReader fileReader = new FileReader(fileName);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;
            while((line = bufferedReader.readLine()) != null) {

                sentences.add(line);

                //Get a list of words
                String[] sent = line.split("\t");
                String[] words = sent[1].split(" ");

                for(String word : words){
                    //System.out.println(word);

                    //Convert to lowercase
                    word = word.toLowerCase();

                    //Remove trailing newline
                    word = word.replace("\n", "").replace("\r", "");

                    //Punctuation
                    if (Pattern.matches("\\p{Punct}", word))
                        continue;

                    //Special character
                    if (word.equals("<<") || word.equals(">>"))
                        continue;

                    //Stop words
                    if (stopwords.contains(word))
                        continue;

                    if (!features.contains(word)){
                        features.add(word);
                        indexes.put(word, features.size()-1);
                    }

                }
            }
            data.features = features;

            bufferedReader.close();

            List<Instance> instances = new ArrayList<Instance>();

            //Generate instances
            for (String sentence : sentences){

                Instance instance = new Instance();

                String[] sent = sentence.split("\t");
                String[] words = sent[1].split(" ");

                int[] feats = new int[features.size()];

                for (int i=0; i<words.length; i++){
                    String word = words[i];
                    if (word.equals(">>")){

                        String target = words[i+1];
                        target = target.toLowerCase();

                        //Set label
                        if (target.equals(word1)){
                            instance.label = 0;
                        }
                        else if (target.equals(word2)){
                            instance.label = 1;
                        }
                        else{
                            throw new NoSuchElementException();
                        }

                        //Add all the words in sentence
                        feats[indexes.get(target)] ++;

                        for (int j = i-1; j>=0 && !isSentenceBreak(words[j]); j--){
                            String f = words[j].toLowerCase();

                            if (indexes.containsKey(f))
                                feats[indexes.get(f)] ++;
                        }

                        for (int j = i+1; j<words.length && !isSentenceBreak(words[j]); j++){
                            String f = words[j].toLowerCase();

                            if (indexes.containsKey(f))
                                feats[indexes.get(f)] ++;
                        }

                        //Add collocation words
                        for (int j = i-1; j>=0 && !isSentenceBreak(words[j]); j--){

                            if (i-j > left_win_size) break;

                            String f = words[j].toLowerCase();

                            if (indexes.containsKey(f))
                                feats[indexes.get(f)] ++;
                        }

                        for (int j = i+1; j<words.length && !isSentenceBreak(words[j]); j++){

                            if (j-i > right_win_size) break;

                            String f = words[j].toLowerCase();

                            if (indexes.containsKey(f))
                                feats[indexes.get(f)] ++;
                        }

                        break;
                    }
                }

                instance.features = feats;
                instances.add(instance);
            }

            data.instances = instances;

        }
        catch(FileNotFoundException ex) {
            System.out.println("Unable to open file '" + fileName + "'");
        }
        catch(IOException ex) {
            System.out.println("Error reading file '" + fileName + "'");
        }


        return data;
    }

    private static List<String> stopwords(){
        List<String> stopwords = new ArrayList<String>();

        try {

            FileReader fileReader = new FileReader("stopwd.txt");
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;
            while((line = bufferedReader.readLine()) != null) {
                line = line.replace("\n", "").replace("\r", "");
                stopwords.add(line);
            }

        }
        catch(FileNotFoundException ex) {
            System.out.println("Unable to open file for stopwords");
        }
        catch(IOException ex) {
            System.out.println("Error reading file for stopwords");
        }

        return stopwords;
    }

    private static boolean isSentenceBreak(String s){
        return s.equals(".") || s.equals("?") || s.equals("!") || s.equals("\"");
    }
}
