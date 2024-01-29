/***
 * Name of File: DecisionTree
 * Creators: Alivia Kliesen, Cheko Mkocheko and Leili Manafi
 * 
 * DecisionTree object represents a binary classification tree composed of TreeNode objects  
 */

import java.util.ArrayList;

public class DecisionTree {
    private TreeNode root = null; //stores the root of the decision tree
	private Integer numTotalFeatures;
	private Integer numFeatChoose;
	private Integer maxDepth;
	private Integer minSamples;

	// Constructor for DecisionTree
	public DecisionTree(int totalFeat, int featChoose, int depth, int minSamp){
		numTotalFeatures = totalFeat;
		numFeatChoose = featChoose;
		maxDepth = depth;
		minSamples = minSamp;
	}
	
	public void train(ArrayList<Example> examples){

		//initialize empty positive and negative lists
		ArrayList<Example> pos = new ArrayList<Example>();
		ArrayList<Example> neg = new ArrayList<Example>();
		
		//paritition examples into positive and negative ones
		for(Example e: examples){
			if (e.getLabel())
				pos.add(e);
			else
				neg.add(e);
		}
		
		//create the root node of the tree
		root = new TreeNode(null, pos, neg, numTotalFeatures);
		root.chooseFeatures(numTotalFeatures, numFeatChoose);
		
		//call recursive train()  on the root node
		train(root);
	}
	
	/**
	 * The recursive train method that builds a tree at TreeNode node
	 * @param node: current node to train
	 * @param numFeatures: total number of features
	 */
	private void train(TreeNode node){
		int numPos = node.pos.size(); // positive examples count
		int numNeg = node.neg.size(); // negative examples count
		int numExamples = numPos + numNeg;
		
		// base case 1
		if(numPos == 0 && numNeg > 0){ // all negative examples
			node.decision = false;
			node.isLeaf = true;
		}
		// base case 2 
		else if(numNeg == 0 && numPos > 0){ // all positive examples
			node.decision = true;
			node.isLeaf = true;
		}
		// base case 3
		else if(numExamples == 0){ // no examples
			int numParentPos = node.parent.pos.size();
			int numParentNeg = node.parent.neg.size();

			if(numParentPos > numParentNeg){ 
				node.decision = true;
			}
			else{
				node.decision = false;
			}

			node.isLeaf = true;
		}
		// base case 4
		else if(numExamples < minSamples){ // exceeded minimum features
			if(numPos > numNeg){
				node.decision = true;
			}
			else{
				node.decision = false;
			}

			node.isLeaf = true;
		}
		// base case 5
		else if(node.getDepth() > maxDepth){  // exceeded maximum depth
			if(numPos > numNeg){
				node.decision = true;
			}
			else{
				node.decision = false;
			}

			node.isLeaf = true;
		}
		
		else{
			double maxInfoGain = 0; 
			int bestFeature = -1;

			node.chooseFeatures(numTotalFeatures, numFeatChoose); // select subsample of features to choose split feature from

			for(int i=0; i<numTotalFeatures; i++){
				if(node.featureUsed(i) != true){ // feature can be used for split feature selection
					double infoGain =  getEntropy(numPos, numNeg) - getRemainingEntropy(i, node);
					if(infoGain > maxInfoGain){ // current feature has higher info gain than previous 
						maxInfoGain = infoGain;
						bestFeature = i;
					}

				}
			}
			
			if(bestFeature != -1){ 
				node.setSplitFeature(bestFeature);
				createChildren(node, numTotalFeatures);
				train(node.trueChild);
				train(node.falseChild);
			}
			else{ // no split feature found which will improve DecisionTree
				node.isLeaf = true;
			}
		}
	}
	
	/**
	 * Creates the true and false children of TreeNode node
	 * @param node: node at which to create children
	 * @param numFeatures: total number of features
	 */
	private void createChildren(TreeNode node, int numFeatures){

		int feature = node.getSplitFeature();

		ArrayList<Example> trueChildPos = new ArrayList<Example>(); // true for split feature, positive label
		ArrayList<Example> trueChildNeg = new ArrayList<Example>(); // true for split feature, negative label
		ArrayList<Example> falseChildPos = new ArrayList<Example>(); // false for split feature, positive label
		ArrayList<Example> falseChildNeg = new ArrayList<Example>(); // false for split feature, negative label

		for(Example e : node.pos){ // all positive label examples
			if (e.getFeatureValue(feature)){ // true on split feature 
				trueChildPos.add(e);
			}
			else{ // false on split feature
				falseChildPos.add(e);
			}
		}
		for(Example e : node.neg){ // all negative label examples
			if (e.getFeatureValue(feature)){ // true on split feature
				trueChildNeg.add(e);
			}
			else{ // false on split feature 
				falseChildNeg.add(e);
			}
		}

		TreeNode tChild = new TreeNode(node, trueChildPos, trueChildNeg, numTotalFeatures);
		TreeNode fChild = new TreeNode(node, falseChildPos, falseChildNeg, numTotalFeatures);
	
		node.trueChild = tChild;
		node.falseChild = fChild;
	}
	
	
	/**
	 * Computes and returns the remaining entropy if feature is chosen
	 * at node.
	 * @param feature: the feature number
	 * @param node: node at which to find remaining entropy
	 * @return remaining entropy at node
	 */
	private double getRemainingEntropy(int feature, TreeNode node){
		//System.out.println("Getting Entropy for node: n with feature" + feature + "with numFeatures: " + numFeatures);

		ArrayList<Example> trueChildPos = new ArrayList<Example>(); // true for split feature, positive label
		ArrayList<Example> trueChildNeg = new ArrayList<Example>(); // true for split feature, negative label
		ArrayList<Example> falseChildPos = new ArrayList<Example>(); // false for split feature, positive label
		ArrayList<Example> falseChildNeg = new ArrayList<Example>(); // false for split feature, negative label

		for(Example e : node.pos){ // all positive label examples
			if (e.getFeatureValue(feature)){ // true on split feature 
				trueChildPos.add(e);
			}
			else{ // false on split feature
				falseChildPos.add(e);
			}
		}
		for(Example e : node.neg){ // all negative label examples
			if (e.getFeatureValue(feature)){ // true on split feature
				trueChildNeg.add(e);
			}
			else{ // false on split feature 
				falseChildNeg.add(e);
			}
		}

		int falseChildSize = falseChildPos.size() + falseChildNeg.size();
		int trueChildSize = trueChildPos.size() + trueChildNeg.size();

		double totalChildren = falseChildSize + trueChildSize;
		
		double falseEntropy = getEntropy(falseChildPos.size(), falseChildNeg.size());
		double trueEntropy = getEntropy(trueChildPos.size(), trueChildNeg.size());
		double remEntropy = (falseChildSize / totalChildren) * falseEntropy + (trueChildSize / totalChildren) * trueEntropy;
		
		return remEntropy;		
	}
	
	/**
	 * Computes the entropy of a node given the number of positive and negative examples it has
	 * @param numPos: number of positive examples
	 * @param numNeg: number of negative examples
	 * @return - entropy
	 */
	private double getEntropy(int numPos, int numNeg){ // works as expected
		double numPosD = numPos;
		double numNegD = numNeg;
		double totalNum = numPosD + numNegD; // total number of examples
		double probPos = numPosD / totalNum; // probability of positive examples
		double probNeg = numNegD / totalNum; // probability of negative examples
		double entropy = 0;
		if(totalNum == 0){ 
			entropy = 0;
		}
		else if(probPos == 0){
			entropy =  (-1) * (probNeg * log2(probNeg));

		}
		else if(probNeg == 0){
			entropy = (-1) * (probPos * log2(probPos));

			}
		else{
			entropy = ((-1) * (probPos * log2(probPos))) - (probNeg * log2(probNeg));
		}
		
		return entropy;
	}
	
	/**	
	 * Computes log_2(d) (To be used by the getEntropy() method)
	 * @param d - value
	 * @return log_2(d)
	 */
	private double log2(double d){
		return Math.log(d)/Math.log(2);
	}
	
	/** 
	 * Classifies example e using the learned decision tree
	 * @param e: example
	 * @return true if e is predicted to be  positive,  false otherwise
	 */
	public boolean classify(Example e){
		TreeNode currNode = root;
		int currFeature = 0;

		while(currNode.isLeaf == false){ 
			currFeature = currNode.getSplitFeature();
			boolean split = e.getFeatureValue(currFeature);
			if(split){ //child evaluates to true
				currNode = currNode.trueChild;
			}
			else{ // child evaluates to false
				currNode = currNode.falseChild;
			}
		}

		return currNode.decision;
	}
	
	
	
	
	//----------DO NOT MODIFY CODE BELOW------------------
	public void print(){
		printTree(root, 0);
	}
	

	
	private void printTree(TreeNode node, int indent){
		if(node== null)
			return;
		if(node.isLeaf){
			if(node.decision)
				System.out.println("Positive");
			else
				System.out.println("Negative");
		}
		else{
			System.out.println();
			doIndents(indent);
			System.out.print("Feature "+node.getSplitFeature() + " = True:" );
			printTree(node.trueChild, indent+1);
			doIndents(indent);
			System.out.print("Feature "+node.getSplitFeature() + " = False:" );//+  "( " + node.falseChild.pos.size() + ", " + node.falseChild.neg.size() + ")");
			printTree(node.falseChild, indent+1);
		}
	}
	
	private void doIndents(int indent){
		for(int i=0; i<indent; i++)
			System.out.print("\t");
	}
}

