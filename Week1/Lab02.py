import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Load the dataset
X_train, X_val, y_val = load_data()


# UNQ_C1
# GRADED FUNCTION: estimate_gaussian

def estimate_gaussian(X):
    """
    Calculates mean and variance of all features
    in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape

    ### START CODE HERE ###
    mu = np.mean(X, axis=0)

    var = np.zeros(n)
    for i in range(m):
       var += (X[i] - mu)**2
    var /= m
    ### END CODE HERE ###
    #print(f" Example entry: {X[0]}")
    #print(f" Average: {mu}")
    #print(f" Standard deviation: {var}")

    return mu, var


def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    # print(f"Y val: {y_val[:5], y_val.shape}")
    # print(f"P val: {p_val[:5], p_val.shape} ")
    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        ### START CODE HERE ###
        predictedValues = np.copy(p_val)
        predictedValues[p_val < epsilon] = 1
        predictedValues[p_val >= epsilon] = 0

        tp = np.sum((predictedValues == 1) & (y_val == 1))
        fp = np.sum((predictedValues == 1) & (y_val == 0))
        fn = np.sum((predictedValues == 0) & (y_val == 1))

        if (tp + fp) == 0 or (tp + fn) == 0:
            continue;

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        F1 = 2 * precision * recall / (precision + recall)
        ### END CODE HERE ###

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1

# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)

print("Mean of each feature:", mu)
print("Variance of each feature:", var)

# UNIT TEST
from public_tests import *

p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)

# UNIT TEST
select_threshold_test(select_threshold)