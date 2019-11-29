'''
Logistic Regression is a variation of Linear Regression, useful when the
observed dependent variable, y, is categorical. It produces a formula
that predicts the probability of the class label as a function of
the independent variables.

Logistic regression fits a special s-shaped curve by taking the
linear regression and transforming the numeric estimate into a
probability with the sigmoid function

The objective of logistic regression is to find the best parameters so that
the model best predicts the class of each case
'''
# import required libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

'''
("ChurnData.csv")
data set includes information about:

Customers who left within the last month – the column is called Churn

Services that each customer has signed up for –
phone, multiple lines, internet, online security, online backup,
device protection, tech support, and streaming TV and movies

Customer account information –
how long they’ve been a customer, contract, payment method,
paperless billing, monthly charges, and total charges

Demographic info about customers – gender, age range, and if
they have partners and dependents
'''
churn_df = pd.read_csv("ChurnData.csv")
print(churn_df.head())

print()

# let's select some features for modeling + we need to change data type to be
# integer, since skitlearn algorithm requires it
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

print()

# let's define X and y for dataset
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5])
y = np.asarray(churn_df['churn'])
print(y[0:5])

print()

# let's normalize the dataset
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

print()

# let's generate train and test data, and confirm matrix sizes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

print()

'''
Lets build our model using LogisticRegression from Scikit-learn package.
This function implements logistic regression and can use different
numerical optimizers to find parameters, including ‘newton-cg’,
‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers.
You can find extensive information about the pros and cons of these
optimizers if you search it in internet.

The version of Logistic Regression in Scikit-learn, support regularization.
Regularization is a technique used to solve the overfitting problem
in machine learning models. 'C' parameter indicates inverse of regularization
strength which must be a positive float.
Smaller values specify stronger regularization
'''
# create model and fit with training data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)

# predict using test data and output probabilities for all classes
yhat = LR.predict(X_test)
print(yhat)
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

print()

# calculate accuracy using jaccard index
from sklearn.metrics import jaccard_similarity_score
print("Jaccard index accuracy:", jaccard_similarity_score(y_test, yhat))

print()

# calculate accuracy using confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
## Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)
## Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
## Print classification report
print (classification_report(y_test, yhat))
print("Confusion matrix accuracy = weighted average f1 score above")
  
print()

# calculate accuracy using log loss
from sklearn.metrics import log_loss
print("Log loss accuracy:", log_loss(y_test, yhat_prob))

print()
