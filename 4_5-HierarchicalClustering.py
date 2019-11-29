'''
Imagine that an automobile manufacturer has developed prototypes for a
new vehicle. Before introducing the new model into its range,
the manufacturer wants to determine which existing vehicles on the
market are most like the prototypes--that is, how vehicles can be grouped,
which group is the most similar with the model, and therefore which models
they will be competing against.

Our objective here, is to use clustering methods, to find the most
distinctive clusters of vehicles. It will summarize the existing vehicles
and help manufacture to make decision about new models simply.
("cars_clus.csv")
'''
# import required libraries
import numpy as np 
import pandas as pd

# read dataset
pdf = pd.read_csv("cars_clus.csv")
print("Shape of dataset: ", pdf.shape)
print(pdf.head(5))

print()

# data cleaning; drop rows that have null values
print("Size of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Size of dataset after cleaning: ", pdf.size)
print(pdf.head(5))

print()

# select featureset
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# normalize data
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
print(feature_mtx[0:5])

print()

'''
CLUSTERING with Scipy
'''
print("...CLUSTERING with Scipy...")
## Clustering using Scipy
# calculate distance matrix
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

# use 'complete' linkage
import pylab
from scipy.cluster import hierarchy 
Z = hierarchy.linkage(D, 'complete')

# use cutting line while clustering (optional)
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

print()

# determine number of clusters directly (optional)
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)

print()

# plot dendrogram
from matplotlib import pyplot as plt 
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
plt.show()
##

'''
CLUSTERING with Scikit-learn
'''
print("...CLUSTERING with Scikit-learn...")
## Clustering using Scikit-learn
# calculate distance matrix
from scipy.spatial import distance_matrix 
dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)

print()

# use 'complete' linkage agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)

# add the new 'cluster' field to dataframe
pdf['cluster_'] = agglom.labels_
print(pdf.head())

# visualize it via scatterplot
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

'''
we are seeing the distribution of each cluster using the scatter plot,
but it is not very clear where is the centroid of each cluster.
Moreover, there are 2 types of vehicles in our dataset,
"truck" and "car" (value of 'type' column). So, we use them to distinguish
the classes, and summarize the cluster
'''
# count 'car' and 'truck' cases in clusters
print(pdf.groupby(['cluster_','type'])['cluster_'].count())
'''
It is obvious that we have 3 main clusters with the majority of
vehicles in those
'''

# let's look at characteristics of each cluster
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
print(agg_cars)
'''
Cars:
Cluster 1: with almost high mpg, and low in horsepower.
Cluster 2: with good mpg and horsepower, but higher price than average.
Cluster 3: with low mpg, high horsepower, highest price.

Trucks:
Cluster 1: with almost highest mpg among trucks, and lowest in horsepower
and price.
Cluster 2: with almost low mpg and medium horsepower, but higher price
than average.
Cluster 3: with good mpg and horsepower, low price.

Notice that we did not use 'type' and 'price' of cars in the clustering
process, but Hierarchical clustering could forge the clusters and
discriminate them with quite high accuracy.
'''

# scatter plot using these characteristics
plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()
##
