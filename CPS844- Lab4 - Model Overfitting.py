import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

# 1) (10 points) Load the data (Y is the class labels of X)
X = np.load(r"C:\Users\Kireh Kaka\Downloads\Xdata.npy")
Y = np.load(r"C:\Users\Kireh Kaka\Downloads\Ydata.npy")

# 2) (15 points) Split the training and test data as follows: 
    # 80% of the data for training and 20% for testing. 
    # Preserve the percentage of samples for each class using the argument 'stratify'. 
    # Use the argument 'random_state' so that the data splitting is the same everytime your code is run.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# 3) (50 points) Test the fit of different decision tree depths 
# Instruction 1: Use the range function to create different depths options, ranging from 1 to 50, for the decision trees
# Instruction 2: As you iterate through the different tree depth options, please:
    # create a new decision tree using the 'max_depth' argument
    # train your tree
    # apply your tree to predict the 'training' and then the 'test' labels
    # compute the training accuracy
    # compute the test accuracy
    # save the training & testing accuracies and tree depth, so that you can use them in the next steps
acc_train = []
acc_test = []
tree_depth = list(range(1, 51))
    
for i in tree_depth:
    clf = tree.DecisionTreeClassifier(max_depth=i,random_state=1)
    clf.fit(X_train, Y_train) 
        
    prediction_train = clf.predict(X_train)  
    prediction_test = clf.predict(X_test)    
        
    atrain = accuracy_score(Y_train, prediction_train)  
    atest = accuracy_score(Y_test, prediction_test)  
        
    acc_train.append(atrain)
    acc_test.append(atest)
    
# 4) (10 points) Plot of training and test accuracies vs the tree depths  
plt.plot(tree_depth, acc_train,'rv-', tree_depth, acc_test,'bo--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.show()

# 5) (15 points) Fill out the following blank:
# Model overfitting happens when the tree depth is greater than __seven___, approximately.