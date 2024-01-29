/***
 * Name of File: GeneralClassifier
 * Creators: Alivia Kliesen, Cheko Mkocheko and Leili Manafi
 * 
 * Reads in data from a training file and a testing file specified by the user and then generates and runs a RandomForest algorithm
 * and outputs accuracy results 
 */

import java.util.*;
import java.io.*;
import java.util.Scanner;

public class GeneralClassifier {

    // Random number generator
    static Random rand = new Random();

	// Examples divided into 4 Arraylists: training (positive and negative) and testing (positive and negative)
	static ArrayList<Example> trainPosWhile, trainNegWhile;
	static ArrayList<Example> testPosWhile, testNegWhile;

	static String trainFileName; // name of file containing Training data
	static String testFileName; // name of file containing Testing data
	static int forestSize;  // # of trees in the Random Forest
	static int numTotalFeat; // total # of features per example 
	static int numFeatChoose; // # of features to choose from at each node for the split feature ( should be less than sqrt(numTotalFeat))
	static int maxTreeDepth; // maximum depth of each decision tree 
    static int minSampSplit; // minimum number of samples at each node needed to split 

    //ArrayList of all training examples (negative and positive)
    static ArrayList<Example> trainExs;

	static Scanner scan = new Scanner(System.in);

    public static void main(String[] args) throws FileNotFoundException {

		boolean analyze = true;
		System.out.println("Welcome to our Random Forest generator. Would you like to create and run a Random Forest on some data?");
		System.out.println("Y / N");
		String response = scan.nextLine();
	
		while(analyze){
			if(response.equals("Y")){ // user wants to generate and run Random Forest algorithm
				getFile();
				getRandomForestParam();
				testClassifier();

				System.out.println("Would you like to create and run another Random Forest?");
				System.out.println("please indicate with Y / N");
				
				String anotherForest = scan.nextLine();
				
				if(anotherForest.equals("N")){ // user is done generating Random Forests
					System.out.println("Goodbye!");
					analyze = false; 
				}
				else if(anotherForest.equals("Y")){ // user wants to continue generating Random Forests
					System.out.println("Awesome! Let's do it...");
					response = "Y";
				}
				else{
					System.out.println("Please enter valid input in the form of'Y' or 'N'. Would you like to create and run a Random Forest?");
				}
			}
			else if(response.equals("N")){ // user does not want to run Random Forest algorithm
				System.out.println("That's too bad! Goodbye! We will miss you...");
				analyze = false; 
			}
			else{ // invalid input from user 
				System.out.println("Please enter valid input in the form of 'Y' or 'N'.");
				response = scan.nextLine();
			}
		}
		scan.close();
		System.exit(0);

        }

	// Gets the names of the training data file and testing data file from the user and stores them as global variables
	public static void getFile(){
		System.out.println("What is the name of the training file you would like to create a random forest for?");
		trainFileName = scan.nextLine(); 

		System.out.println("What is the name of the testing file you would like to run the random forest on?");
		testFileName = scan.nextLine(); 
	}

	// Gets the following parameters for the Random Forest from the user: forestSize, numFeatChoose, maxTreeDepth, minSampleSplit
	// Parameters are stored as global variables
	public static void getRandomForestParam(){
		System.out.println("What size forest would you like to implement? (range 30 - 300)");
		forestSize = scan.nextInt();
		scan.nextLine();
		
		System.out.println("How many features would you like each decision tree in the forest to choose from?");
		numFeatChoose = scan.nextInt();
		scan.nextLine();

		System.out.println("What is the maximum tree depth you would like to allow?");
		maxTreeDepth = scan.nextInt();
		scan.nextLine();

		System.out.println("What is the minimum number of samples needed to split at a node?");
		minSampSplit = scan.nextInt();
		scan.nextLine();
	}
	
	
    // Train and tests a random forest on data 
	// Outputs positive accuracy, negative accuracy, total accuracy, and OOB results 
    private static void testClassifier() throws FileNotFoundException {

		// Initializes empty ArrayLists to store negatie and positive training data 
		trainPosWhile = new ArrayList<Example>();
		trainNegWhile = new ArrayList<Example>();

    	// Load training examples from input training file specified by user previously 
    	loadExamples(trainFileName, trainPosWhile, trainNegWhile);

	    trainExs = new ArrayList<Example>(); // examples for training (positive and negative)

	    //Loading TrainExample array - Positive examples
	    for(Example e : trainPosWhile) {
	    	trainExs.add(e);
	    }

	    //Loading TrainExample array - Negative examples 
	    for(Example e : trainNegWhile) {
	    	trainExs.add(e);
	    }

	    // Create and train the random forest using the training examples and user-specified Random Forest parameters 
		RandomForest randForest = new RandomForest(trainExs, forestSize, numTotalFeat, numFeatChoose, maxTreeDepth, minSampSplit);
	  	randForest.trainForest();

		// Initializes empty ArrayLists to store negative and positive testing data
		testPosWhile = new ArrayList<Example>();
		testNegWhile = new ArrayList<Example>();

		//Load testing examples from input testing file specified by user previously 
		loadExamples(testFileName, testPosWhile, testNegWhile);

		// Evaluate Random Forest classification of all positive testing examples 
		int posCorrect = 0;
		for (Example e : testPosWhile ) {
		    if (randForest.evaluateExample(e)) // algorithm correctly classifies positive example
			posCorrect++;
		}

		System.out.println("Positive examples correct: "+posCorrect+" out of "+ testPosWhile.size());
		double accuracy = Double.valueOf(posCorrect) / Double.valueOf(testPosWhile.size());
		System.out.println("Positive accuracy: " + accuracy );

		// Evaluate Random Forest classification of all negative testing examples
		int negCorrect = 0;
		for (Example e : testNegWhile) {
		    if (!randForest.evaluateExample(e)) // algorithm correctly classifies negative example
			negCorrect++;
		}

		System.out.println("Negative examples correct: "+negCorrect+" out of "+testNegWhile.size());
		accuracy = Double.valueOf(negCorrect) / Double.valueOf(testNegWhile.size());
		System.out.println("Negative accuracy: " + accuracy );
		System.out.println();

		// Calculates and prints out overall accuracy from Random Forest algorithm
		double totalAccuracy =(  Double.valueOf(negCorrect) + Double.valueOf(posCorrect) );
		totalAccuracy = totalAccuracy / (Double.valueOf(testPosWhile.size()) + Double.valueOf(testNegWhile.size()));
		System.out.println("Overall total accuracy is: " + totalAccuracy);

		// Calcualtes and prints out OOB (out of bag) error estimate 
		double score = randForest.calcMeanOobScore();
		System.out.println("OOB: " + score);
    }

	// Loads in examples from a data file and stores the examples in a global Positive Example ArrayList and a global Negative Example ArrayList
	private static void loadExamples(String file, ArrayList<Example> posWhile, ArrayList<Example> negWhile) throws FileNotFoundException
    {
    	Scanner scanner = new Scanner(new File(file));
		numTotalFeat = (scanner.nextLine().split("	").length) - 1 ; // counts number of features in header minus label feature

		while(scanner.hasNextLine()){ // an example exists to be read in 
			String n = scanner.next(); // example label 
			
			if (n.equals("positive")){ 
				Example currExample = new Example(numTotalFeat);

				for(int j=0; j<numTotalFeat; j++){ // iterates through all feature values for example 
	    			if(scanner.hasNextBoolean()){
	    				boolean b = scanner.nextBoolean();
	    				currExample.setFeatureValue(j, b);
	    			}
	    		}
				currExample.setLabel(true);
				posWhile.add(currExample); // adds example to list of positive examples
			}
			else if (n.equals("negative")){ 
				Example currExample = new Example(numTotalFeat);

				for(int j=0; j<numTotalFeat; j++){
	    			if(scanner.hasNextBoolean()){
	    				boolean b = scanner.nextBoolean();
	    				currExample.setFeatureValue(j, b);
	    			}
	    		}
				currExample.setLabel(false);
				negWhile.add(currExample); // adds examples to list of negative examples
			}
			else{
    			System.out.println("ERROR: "+ n); // debugging (shouldn't execute if data is formatted correctly)
    		}
		}
		scanner.close();
	}
}
