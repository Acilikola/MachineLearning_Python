'''
SVM works by mapping data to a high-dimensional feature space so
that data points can be categorized, even when the data are not
otherwise linearly separable. A separator between the categories is found,
then the data are transformed in such a way that the separator
could be drawn as a hyperplane. Following this, characteristics of new data
can be used to predict the group to which a new record should belong

in this example, we will build and train a SVM model using human cell records,
and classify cells to whether the samples are benign or malignant
("cell_samples.csv")

The fields in each record are:

Field name	Description
ID	Clump thickness
Clump	Clump thickness
UnifSize	Uniformity of cell size
UnifShape	Uniformity of cell shape
MargAdh	Marginal adhesion
SingEpiSize	Single epithelial cell size
BareNuc	Bare nuclei
BlandChrom	Bland chromatin
NormNucl	Normal nucleoli
Mit	Mitoses
Class	Benign or malignant

The ID field contains the patient identifiers.
The characteristics of the cell samples from each patient are contained
in fields 'Clump' to 'Mit'. The values are graded from 1 to 10,
with 1 being the closest to benign.

The Class field contains the diagnosis, as confirmed by separate medical
procedures, as to whether the samples are benign (value = 2) or
malignant (value = 4).
'''
# import required libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head())

print()

# check distro of classes based on Clump thickness and Uniformity of cell size
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

# check column data types
print(cell_df.dtypes)

print()

# since 'BareNuc' is not numerical, let's drop those rows
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

print()

# create features array X
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])

# create target as y (benign=2) (malignant=4)
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print(y[0:5])

print()

# generate training and test sets from dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

print()

# let's use the default transformation function, Radial Basis Function (RBF),
# to create our SVM model and fit training data
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
print(clf)

print()

# predict new values using test
yhat = clf.predict(X_test)
print(yhat[0:5])

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

print()

# calculate f1_score
from sklearn.metrics import f1_score
print("F1 score:",f1_score(y_test, yhat, average='weighted'))

print()

# calculate accuracy using jaccard index
from sklearn.metrics import jaccard_similarity_score
print("Jaccard index:",jaccard_similarity_score(y_test, yhat))
