import numpy as np

class MyScale:
    """Base class for Scaler"""
    def fit(self, X):
        raise NotImplementedError

            
    def transform(self, X):
        raise NotImplementedError


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MyStandardScaler(MyScale):
    """Z-score"""
    def fit(self, X, y = None):
        self.mean_ = np.mean(X, axis = 0)
        self.std_ = np.std(X, axis = 0)
        # avoid std = 0
        self.std_[self.std_ == 0] = 1e-9
        return self

    def transform(self, X):
        return (X - self.mean_)/self.std_

class MyMinMaxScaler(MyScale):
    """Min Max Scaler"""
    def fit(self, X, y = None):
        self.max_ = np.max(X, axis=0)
        self.min_ = np.min(X, axis=0)

        # avoid min == max 
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1e-9
        return self
    

    def transform(self, X):
        return (X - self.max_) / (self.range_)
