# to cluster my data in categories
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# in here(especially k-means) the last column is not the depending variable it is a feature
# wich we will use as part of the other feature to identify some paterns(clusters /segment)
'''
in this case for learning purposes we will remove the age column to become 2D plot to be able 
to visualise it 
'''
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
'''
using the elbow method to find the optimal number of clusters
after seeing the graph we will choose manually the best value for the cluster
'''

# we will do a loop run k-means algortithms with 10 values of clusters to find the best one
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''
training the k-means model on the dataset & build the dependant variable 
'''
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans_pred = kmeans.fit_predict(x)
'''
visualising the clusters
'''
plt.scatter(x[y_kmeans_pred == 0, 0], x[y_kmeans_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans_pred == 1, 0], x[y_kmeans_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans_pred == 2, 0], x[y_kmeans_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans_pred == 3, 0], x[y_kmeans_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans_pred == 4, 0], x[y_kmeans_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()