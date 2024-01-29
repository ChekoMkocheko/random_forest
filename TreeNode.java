/***
 * Name of File: TreeNode
 * Creators: Alivia Kliesen, Cheko Mkocheko and Leili Manafi
 * 
 * TreeNode object represents a single node on a DecisionTree 
 */


import java.util.*;

public class TreeNode {
    TreeNode parent; //parent of this node
	TreeNode trueChild;//the examples which are true on the splitFeature
	TreeNode falseChild;//the examples which are true on the splitFeature
	
	ArrayList<Example> pos; // the positive examples at this node
	ArrayList<Example> neg; //the negative examples at this node
	boolean decision;
	int depth; 
	
	private boolean[] featuresUsed; // the features already used prior to this node 
	private int splitFeature;// the feature that this node will split examples on
	boolean isLeaf = false; //indicates whether the node is a leaf (set to true during pruning)
	
	static Random rand = new Random();
	
	// Constructor for TreeNode object 
	public TreeNode(TreeNode par, ArrayList<Example> p, ArrayList<Example> n, int numTotalFeat){
		parent = par;
		resetFeatures(numTotalFeat); 

		pos = p;
		neg = n;
		if(pos.size()==0 || neg.size() ==0)
			isLeaf = true;
		splitFeature = -1;
		setDepth();
	}

	// randomly chooses indices of features to be used by a given node for split feature selection
	public void chooseFeatures(int numTotalFeat, int numFeatUse) {
		ArrayList<Integer> randomFeatures = new ArrayList<Integer>();
        
        for(int j = 0; j < numFeatUse; j++){
            int randFeature = rand.nextInt(numTotalFeat);
            while(randomFeatures.contains(randFeature)){ // feature cannot already be selected 
                randFeature = rand.nextInt(numTotalFeat); // choose feature that hasn't been selected already for this node 
            }
            randomFeatures.add(randFeature);
        }
		
		for(int i=0; i<randomFeatures.size(); i++){
			int index = randomFeatures.get(i);
			featuresUsed[index] = false;
		}
	}
	
		
	/**
	 * Reset the featuresUsed array to all true values 
	 */
	private void resetFeatures(int features){
		featuresUsed = new boolean[features];
		
		for(int i =0; i<featuresUsed.length; i++){
			featuresUsed[i]= true;
		}
	}

	// Initializes depth of the node based on whether it is the root or its parents depth 
	private void setDepth(){
		if(parent == null){ // node is root 
			depth = 0;
		}
		else { // node is not root 
			depth = parent.depth + 1;
		}
	}

	// Getter method for depth 
	public int getDepth(){
		return this.depth;
	}
	/**
	 * returns whether or not the input feature i can be used in this tree
	 * @param i - the feature
	 * @return - true if i cannot be used for split feature selection and false otherwise
	 */
	public boolean featureUsed(int i){
		return featuresUsed[i];
	}
	
	/**
	 * Set feature f to be the one to split this node on 
	 * @param f - feature
	 */
	public void setSplitFeature(int f){
		splitFeature = f;
		featuresUsed[f] = true;
	}
	
	/** return the feature this node is split with 
	 * @return - index of split feature
	 */
	public int getSplitFeature(){
		return splitFeature;
	}
	
	public String toString(){
		return splitFeature + " \t " + pos.size() + " \t " + neg.size() + "\t " + parent.splitFeature;
	}

}
