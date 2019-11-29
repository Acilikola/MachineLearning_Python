'''
K-means is vastly used for clustering in many data science applications,
especially useful if you need to quickly discover insights from unlabeled data.
In this notebook, you learn how to use k-Means for customer segmentation.

In this notebook we practice k-means clustering with customer segmentation

Imagine that you have a customer dataset, and you need to apply customer
segmentation on this historical data. Customer segmentation is the practice
of partitioning a customer base into groups of individuals that have similar
characteristics
("Cust_Segmentation.csv")
'''
# load data
import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.head())

print()

# since 'Address' is a categorical variable and not compatible with k-means
# clustering (can't calculate Euclidean distance), let's drop it
df = cust_df.drop('Address', axis=1)
print(df.head())

print()

# let's normalize the dataset to treat all features equally
import numpy as np
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

print()

# let's apply k-means on our dataset and check labels
from sklearn.cluster import KMeans
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

print()

# assign labels to each row in dataframe
df["Clus_km"] = labels
print(df.head(5))

print()

# check centroid values by averaging features in each cluster
print(df.groupby('Clus_km').mean())

print()

# 2D visualization, distro of customers based on age and income
import matplotlib.pyplot as plt
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

## 3D visualization, distro of customers based on age, income, education
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()

'''
The customers in each cluster are similar to each other demographically.
Now we can create a profile for each group, considering the common
characteristics of each cluster. For example, the 3 clusters can be:

AFFLUENT, EDUCATED AND OLD AGED
MIDDLE AGED AND MIDDLE INCOME
YOUNG AND LOW INCOME
'''
