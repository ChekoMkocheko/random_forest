/***
* Name of File: RandomForest
 * Creators: Alivia Kliesen, Cheko Mkocheko and Leili Manafi
 * 
 * RandomForest object represents a collection of DecisionTrees that are generated in trainForest() method 
 */

import java.util.*;
import java.io.*;

public class RandomForest {

    private ArrayList<DecisionTree> trees = new ArrayList<DecisionTree>(); 
    private ArrayList<Double> outOfBagErrors = new ArrayList<Double>(); // stores each OOB error calculated from a decision tree in the forest 
    static Random rand = new Random();

    private ArrayList<Example> examples = new ArrayList<Example>(); // stores all the examples used for training the RandomForest

    // parameters for RandomForest object that are initialized in GeneralClassifier.java by user 
    private final int numTrees;
    private final int maxNumFeatures; // should be equal to or less than square root of total number of features
    private final int numTotalFeatures;
    private final int maxTreeDepth;
    private final int minSamplesSplit;

    // Constructor for RandomForest object
    public RandomForest(ArrayList<Example> ex, int size, int numTotalFeat, int maxNumFeat, int treeDepth, int minSamp) {
        examples = ex;
        numTrees = size;
        numTotalFeatures = numTotalFeat;
        maxNumFeatures = maxNumFeat;
        maxTreeDepth = treeDepth;
        minSamplesSplit = minSamp;
    }

    // Trains the RandomForest on the training examples 
    public void trainForest() {

        for (int i=0; i<numTrees; i++){ // one tree from the forest at a time
            ArrayList<Example> bootstrapEx = bootstrap(); // subsamples training data using bootstrapping 

            DecisionTree tree = new DecisionTree(numTotalFeatures, maxNumFeatures, maxTreeDepth, minSamplesSplit); 
            tree.train(bootstrapEx); 
            trees.add(tree);
            ArrayList<Example> oobEx = getOobEx(bootstrapEx); // stores OOB examples not subsampled during bootstrapping
            double oobScore = calcOobScore(oobEx, tree); // calculates OOB error using OOB examples 
            outOfBagErrors.add(oobScore); 
        }
    }

    // Evaluates an example using the trained RandomFOrest object 
    // Returns the majority classification (positive or negative) of all the trees in the forest 
    public Boolean evaluateExample(Example ex) {
        int numTrue = 0;
        int numFalse = 0;
        boolean finalAnswer = true;

        for(int i=0; i<numTrees; i++){
            DecisionTree tree = trees.get(i);
            Boolean answer = tree.classify(ex);
            if(answer == true){ // DecisionTree classifies example as positive 
                numTrue++;
            }
            else{ // DecisionTree classifies example as negative
                numFalse++;
            }
        }
        if(numFalse > numTrue){ // majority of forest classifies example as negative 
            finalAnswer = false; // update finalAnswer (initialized as true) to false 
        }
        return finalAnswer;
    }

     // calculates the average out of bag error score for a random forest 
     public double calcMeanOobScore() {
        int n = outOfBagErrors.size(); // should be same as numTrees
        double sum = 0;

        for(int i=0; i<n; i++){
            double score = outOfBagErrors.get(i);
            sum+=score;
        }

        return sum / Double.valueOf(n); // calculates average of all OOB scores from forest 
    }

    // Randomly generate bootstrap examples
    private ArrayList<Example> bootstrap() {
        ArrayList<Example> bootstrapExamples = new ArrayList<Example>();
        int n = examples.size();
        for(int i = 0; i < n; i++){
            int index = rand.nextInt(n); // random index (replacement allowed in bootstrapping)
            Example ex = examples.get(index);
            bootstrapExamples.add(ex);
        }
        return bootstrapExamples;
    }
    
    // get out-of-bag samples for a given tree
    private ArrayList<Example> getOobEx(ArrayList<Example> bootstrapEx) {
        ArrayList<Example> oobData = new ArrayList<Example>();
        int n = examples.size();
        for(int i=0; i<n; i++){
            Example currEx = examples.get(i);
            if(!bootstrapEx.contains(currEx)){ // current example not in bootstrapped subsample of all examples 
                oobData.add(currEx); // add current example to list of OOB examples for later calculation of OOB error
            }
        }
        return oobData;
    }

    // Calculates the out of bag score for a single decision tree
    private double calcOobScore(ArrayList<Example> oobData, DecisionTree tree){
        double numIncorrect = 0;
        int n = oobData.size();
        for(int i=0; i<n; i++){
            Example currEx = oobData.get(i);
            boolean pred = tree.classify(currEx); 
            boolean actual = currEx.getLabel();

            if(pred != actual){ // DecisionTree incorrectly classifies the OOB example 
                numIncorrect+=1.0;
            }
        }
        return numIncorrect / Double.valueOf(n); // percentage of incorrect OOB examples 
    }

}