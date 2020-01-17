import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def insert_intercept(dataframe):
    dataframe.insert(1, 'intercept', 1)
    return dataframe

def split_data(dataframe):

    np_data = dataframe.values
    X = np_data[:,1:]
    y = np_data[:,0]

    assert type(X) == np.ndarray
    assert type(y) == np.ndarray

    return X, y

def CoordinateLasso(X, y, lambda_):
    np.random.seed(0)
    training_error_history = []
    
    N = X.shape[0]
    D = X.shape[1]
    w = np.random.normal(0, 1, D) # initialize w

    z = []
    for j in range(0, D):
        z.append(X[:,j].T.dot(X[:,j])) # pre-compute z_j
    
    while(True):
        w_prev = w.copy()
        
        for j in range(0, D):
            
            p_j = (y - X.dot(w.T) + X[:,j] * w[j]).dot(X[:,j]) # compute p_j
                                  
            if not (j == 0): # apply soft-thresholding, don't penalize intercept                 
                if (p_j < -lambda_/2):
                    p_j = p_j + lambda_/2
                elif (p_j > lambda_/2):
                    p_j = p_j - lambda_/2
                else:
                    p_j = 0

            w[j] = p_j / z[j] # update w_j
        
        residual = y - X.dot(w.T)                          
        training_error_history.append(residual.T.dot(residual))
        if (max(abs(w_prev - w)) < 10**-6): break
            
    return w, training_error_history

def plot_error_over_iterations(error_history):
    
    fig = plt.figure(figsize =(30,7))
    
    fontsize = 30
    
    x = list(range(len(error_history)))
    y = error_history
    
    plt.plot(x, y, marker = '*', markersize = 20)
    
    plt.title('Training error over iterations', {'fontsize': fontsize})
    plt.ylabel('Squared errors', {'fontsize': fontsize})
    plt.xlabel('Iterations', {'fontsize': fontsize})
    
    return fig
        
def stack_weights_by_lambda(lambda_ , X, y):
    
    w_tot = []
    for l in lambda_:
#         print('Training for lambda = ' + str(l))
        w_tot.append(CoordinateLasso(X, y, l)[0].tolist())
        
    w_tot = np.array(w_tot)
    
    assert w_tot.shape == (10,96)
    return w_tot
    
    
def plot_weights(lambda_, w_tot, dataframe, features):
    
    fig = plt.figure(figsize=(30,7))
    fontsize = 30
    
    log_lambda = np.log(lambda_)
    
    w_extracted = np.array([w_tot[:, dataframe.columns.get_loc(feature) - 1] for feature in features])
#     w_extracted = np.array([w_tot[:, j] for j in range(0, 96)]) # uncomment this and below to see all features
    
    plt.xlim(max(np.log(lambda_)), 0)
    
#     for i in range(0, 96): plt.plot(log_lambda, w_extracted[i], label = dataframe.columns[i+1], marker = '*', markersize = 20)
    plt.plot(log_lambda, w_extracted[0], label = features[0], marker = '*', markersize = 20)
    plt.plot(log_lambda, w_extracted[1], label = features[1], marker = '*', markersize = 20)
    plt.plot(log_lambda, w_extracted[2], label = features[2], marker = '*', markersize = 20)
    plt.plot(log_lambda, w_extracted[3], label = features[3], marker = '*', markersize = 20)
    plt.plot(log_lambda, w_extracted[4], label = features[4], marker = '*', markersize = 20)
    
    plt.title('Regularization paths', {'fontsize': fontsize})
    plt.ylabel('Weights', {'fontsize': fontsize})
    plt.xlabel('log($\lambda$)', {'fontsize': fontsize})
    plt.legend(fontsize = 20)

    return fig
      
    
def plot_training_error(lambda_, w_tot,  X, y):
    
    fig = plt.figure(figsize=(30,7))
    
    w_tot = np.flip(w_tot, 0)
    sqr_errors = []
    
    for w in w_tot:
        residual = y-X.dot(w.T)
        sqr_errors.append(residual.dot(residual.T))
    
    plt.plot(np.flip(np.log(lambda_)), sqr_errors, marker = '*', markersize = 20)
    plt.title('Training errors over log($\lambda$)', {'fontsize': 30})
    plt.ylabel('Squared errors ', {'fontsize': 30})
    plt.xlabel('log($\lambda$)', {'fontsize': 30})
    
    return fig


def plot_test_error(lambda_, w_tot,  X, y):
    
    fig = plt.figure(figsize=(30,7))
    
    w_tot = np.flip(w_tot, 0)
    sqr_errors = []
    
    for w in w_tot:
        residual = y-X.dot(w.T)
        sqr_errors.append(residual.dot(residual.T))
    
    plt.plot(np.flip(np.log(lambda_)), sqr_errors, marker = '*', markersize = 20)
    plt.title('Test errors over log($\lambda$)', {'fontsize': 30})
    plt.ylabel('Squared errors', {'fontsize': 30})
    plt.xlabel('log($\lambda$)', {'fontsize': 30})

    return fig

def plot_number_of_nonzero_index(lambda_, w_tot):
    
    fig = plt.figure(figsize=(30,7))
    
    num_non_zeros = [np.count_nonzero(w) for w in w_tot]
    
    plt.plot(np.flip(lambda_), np.flip(num_non_zeros), marker = '*', markersize = 20)
    plt.title('Number of non-zero weights', {'fontsize': 30})
    plt.ylabel('Number of non-zero weights', {'fontsize': 30})
    plt.xlabel('$\lambda$', {'fontsize': 30})
    
    return fig