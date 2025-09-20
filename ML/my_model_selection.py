import numpy as np


def my_train_test_split(X, y, test_size = 0.2, random_state = None):
    """split the train set and test set"""
    if random_state:
        np.random.seed(random_state)


    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # shuffle 
    shuffle_indices = np.random.permutation(n_samples)

    test_indice = shuffle_indices[:n_test]
    train_indice = shuffle_indices[n_test:]

    X_train, X_test = X[train_indice], X[test_indice]
    y_train, y_test = y[train_indice], y[test_indice]

    return X_train, X_test, y_train, y_test

