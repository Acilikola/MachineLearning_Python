#import required libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

'''
("teleCust1000t.csv")
example focuses on using demographic data, such as region, age, and marital,
to predict usage patterns.
The target field, called custcat, has four possible values that
correspond to the four customer groups, as follows:
1- Basic Service 2- E-Service 3- Plus Service 4- Total Service

Our objective is to build a classifier, to predict the class of unknown cases.
We will use a specific type of classification called K nearest neighbour.
'''

df = pd.read_csv("teleCust1000t.csv")
print(df.head())

# let's see how many of each class is in our dataset
print(df['custcat'].value_counts())

# explore data using visualization
plt.hist(df['income'], bins=50)
plt.show()

# check column names
print(df.columns)

# to use scikit-learn library, need to convert Pandas data frame to Numpy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5])
# and our lables
y = df['custcat'].values
print(y[0:5])

# data normalization(standardization) give data zero mean and unit variance,
# which is a good practice especially for algorithms such as KNN based on distance
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

# generate train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# import knn classifier
from sklearn.neighbors import KNeighborsClassifier

# and start algorithm with k=4 initially
k = 4
##Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

##Predict test set with model
yhat = neigh.predict(X_test)
print(yhat[0:5])

# accuracy evaluation
'''
In multilabel classification, accuracy classification score function
computes subset accuracy. This function is equal to the
jaccard_similarity_score function. Essentially, it calculates
how match the actual labels and predicted labels are in the test set.
'''
from sklearn import metrics
print("Train set Accuracy k=4: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy k=4: ", metrics.accuracy_score(y_test, yhat))

print()

# build and test the model again, with k=6
k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy k=6: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy k=6: ", metrics.accuracy_score(y_test, yhat6))

print()

'''
Proper KNN model generation, test and choosing the best option
'''
# KNN generally starts with k=1 and repeatedly builds a new model and
# tests accuracy, increasing k in each iteration
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

# plot model accuracy for different number of neighbors, k
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

# choose most accurate / best K value
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
