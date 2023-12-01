""" Clustering pipeline -DT 11/23/21
The intention is to add clustering methods for the following:
1. Partitional clustering (KMeans)
2. Hierarchical clustering (Agglomerative)
3. Density-based (TBD) """

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage


""" The data passed to plot_KMeans_knee is assumed to be already preprocessed
    knee curve shape is assumed to be convex and decreasing."""


def plot_kmeans_knee(data, max_k=11, return_sse=False):
    kmeans_kwargs = {
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
    }
    sse = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    k = np.arange(1, max_k)
    kl = KneeLocator(k, sse, curve='convex', direction='decreasing')
    kl.plot_knee()
    plt.xlabel('k')
    plt.xticks(k)
    plt.ylabel('sse')
    plt.show()
    print('KneeLocator elbow: '+str(kl.elbow))
    if (return_sse):
        return k, sse
    else:
        return


def plot_kmeans_silhouette_coeff(data, max_k=11, return_SC=False):
    kmeans_kwargs = {
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
    }
    sc = []  # silhouette coefficient requires >=2 clusters
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        sc.append(score)
    k = np.arange(2, max_k)
    n = k[sc.index(max(sc))]
    plt.plot(k, sc, color='b')
    plt.axvline(x=n, color='tab:blue', linestyle='--')
    plt.xlabel('k')
    plt.xticks(k)
    plt.ylabel('Silhouette Coefficient')
    plt.show()
    print('No. of clusters to choose is the maximum silhouette coefficient.')
    print('Max Silhouette Coefficient k=' +
          str(k[sc.index(max(sc))]))
    if (return_SC):
        return k, sc
    else:
        return


""" The dendrogram is generated using the Agglomerative clustering method
    using the default 'ward' linkage method."""


def plot_dendrogram(Z, max_d=None):
    plt.xlabel('sample index')
    plt.ylabel('distance')
    if max_d is None:
        dendrogram(Z, no_labels=True)
        plt.show()
    elif type(max_d) is float:
        dendrogram(Z, no_labels=True)
        plt.axhline(y=max_d, linestyle='--')
        plt.show()
    else:
        print('max_d error.')
        dendrogram(Z, no_labels=True)
        plt.show()


def plot_trunc_dendrogram(Z, p=5):
    plt.figure()
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z,
               truncate_mode='lastp',
               p=p,  # Only show final p merges (horizontal lines)
               leaf_rotation=90.0,
               leaf_font_size=8.0)
    plt.show()


def get_Z_ward(data):
    # Calculate the linkage between data points
    Z = linkage(data, 'ward')  # the ward method is bottom -> up euclidean
    # The format of Z; rows: n-1, columns: indx1, indx2, dist, n_individuals
    return(Z)


def get_clstrs_dist(Z, max_d):
    return(fcluster(Z, max_d=max_d, criterion='distance'))


def get_clstrs_k(Z, max_k):
    return(fcluster(Z, max_k, criterion='maxclust'))
