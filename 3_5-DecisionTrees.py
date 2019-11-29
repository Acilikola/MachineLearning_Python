'''
learn a popular machine learning algorithm, Decision Tree.
You will use this classification algorithm to build a model from
historical data of patients, and their respond to different medications.
Then you use the trained decision tree to predict the class of a
unknown patient, or to find a proper drug for a new patient
("drug200.csv")
'''
# import required libraries
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# read dataset
my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])

print()

# let's analyze dataset
##how many of each class is in our dataset
print(my_data['Drug'].value_counts())

print()

##check column names
print(my_data.columns)

print()

# create feature matrix as X
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

print()

# since some features (sex, BP) are categorical and Sklearn Decision Trees
# do not support them, we need to convert categorical features to numerical
# values
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

print()

# create target variable as y
y = my_data['Drug']
print(y[0:5])

print()

'''
We will be using train/test split on our decision tree.

train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset

The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.

The X and y are the arrays required before the split,
the test_size represents the ratio of the testing dataset,
and the random_state ensures that we obtain the same splits
'''
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# print the shapes of X_trainset and y_trainset, X_testset and y_testset
# to ensure dimensions match
print('Train set:', X_trainset.shape,  y_trainset.shape)
print('Test set:', X_testset.shape,  y_testset.shape)

print()

# We will first create an instance of the DecisionTreeClassifier, 'drugTree'
# Inside of the classifier, specify criterion="entropy"
# so we can see the information gain of each node
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree)

# fit the training data to tree and make predictions and store it in 'predTree'
drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)

# import metrics from sklearn and check accuracy of model
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# let's visualize the tree
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
