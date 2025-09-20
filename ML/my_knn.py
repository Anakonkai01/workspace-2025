import numpy as np
from collections import Counter 


class MyNeighborsClassifier:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distane(self, x1 ,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def _predict_one(self, x):
        distance = [self._euclidean_distane(x, )]
    