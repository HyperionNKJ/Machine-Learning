import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

class kmeans:
    def __init__(self, x1, x2, k):
        self.x1 = x1
        self.x2 = x2
        self.k = k
        self.X = np.array(list(zip(x1, x2)))
    
    # return X, cluster labels, coordinates of cluster centers(shape = (15,2))
    def clustering(self):
        
        # initial cluster centers
        np.random.seed(0)
        
        # x coordinates of random cluster center
        C_x = np.random.randint(0, np.max(self.x1)-np.mean(self.x1), size=self.k)
        # y coordinates of random cluster center
        C_y = np.random.randint(0, np.max(self.x2)-np.mean(self.x2), size=self.k)
        self.C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
        
        self.cluster_labels = np.zeros(self.X.shape[0], dtype=int) # initialize labels to all 0

        while(True):
            old_C = self.C.copy()
            self.assign_labels()
            self.revise_centroids()
            if (np.array_equal(old_C, self.C)): break
        
        assert self.cluster_labels.shape == (self.X.shape[0],)
        assert self.C.shape == (self.k,2)
        
        return self.X, self.cluster_labels, self.C

     # Euclidean distance
    def EuclideanDistance(self, a, b, ax = 1):
        distance = np.linalg.norm(a-b)
        return distance
    
    def assign_labels(self):
        for i in range(self.X.shape[0]):
            dists = np.array([self.EuclideanDistance(self.X[i], cc) for cc in self.C])
            self.cluster_labels[i] = np.argmin(dists) # use index of centroids as labels
    
    def revise_centroids(self):
        for j in range(self.k):
            self.C[j] = self.X[self.cluster_labels == j].mean(axis=0)
       
    def cluster_heterogeneity(self):
        heterogeneity = 0
        for j in range(self.k):
            members = self.X[self.cluster_labels == j]
            for member in members:
                heterogeneity += self.EuclideanDistance(member, self.C[j])**2
        
        return heterogeneity

def plot_data(X, cluster_labels, C, k):
    fig = plt.figure(figsize=(10,5))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    label_color = [colors[label] for label in cluster_labels]
    plt.scatter(X[:,0], X[:,1], s=1, c=label_color)
    plt.scatter(C[:,0], C[:,1], marker = '*', s = 700, c='k')
    
    return plt