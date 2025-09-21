import numpy as np



class MultinomialNaiveBayes:
    
    def __init__(self, alpha = 1.0):
        self.alpha = alpha # for Laplace smoothing
        
    def fit(self, X, y):
        """
            X: np.ndarray()
            y: np.ndarray()
        """
        
        n_samples, n_features = X.shape 
        self._classes = np.unique(y) 
        n_classes = len(self._classes)
        
        self._priors = 

        
        
        